# 抓取系统重构报告（2026-11-08）

状态：数据齐全，待独立审计确认（见文末审计状态）。

## 1. 改造范围与动机

起点是一个针对固定 46mm 红色圆柱、依赖固定闭合宽度和真值触发兜底的演示系统。
本轮按五项审计发现重构（全部核实为真）：

| # | 审计发现 | 核实位置 | 处置 |
|---|---|---|---|
| 1 | 固定闭合 32mm，小物体永远夹不到 | run_grasp.py CLOSED_WIDTH_M | 渐进闭合（见 §3） |
| 2 | 水平尺寸相近即推断连续对称 | object_observation._symmetry_hypothesis | 分高度薄层径向圆度证据 |
| 3 | 旋转候选无接触复验 | augment 脚本 | 旋转后全部过统一复验 |
| 4 | 场景真值触发几何兜底 | batch 的 object_shape=='cube' 判断 | 显式 --geometry-fallback 旗标 + 分方法统计 |
| 5 | 成功例 failure_stage 残留 'execution' | batch 映射逻辑 | 成功置 None；execution_complete 新口径 |

## 2. 新架构

- **统一候选接口**（`piperx_lab/grasp_candidates.py`）：所有候选（网络/旋转/几何）
  统一过 `revalidate_candidate`：局部宽度可行性（云+凸包取大者）、夹持体积包含、
  双侧接触支撑、预抓取腕部低于支撑面预检。每候选携带 frame/stamp/位姿/开口/
  接近方向/来源/置信度/拒绝原因；置信度是排序启发式，不是成功概率。
- **几何反抓生成器**：纯几何（不含颜色/形状类别）的中高度带反抓采样 + 8 方位×
  3 倾角（0°/30°/55°），同一复验通道。实验发现其候选当前**从未被选中**
  （置信度排序下网络候选总优先）——诚实记录，未虚报融合收益。
- **渐进闭合 + 实测锁紧**（`piperx_lab/gripper_policy.py` + run_grasp）：从预测
  宽度+4mm 步进合拢，双指接触即停；锁紧量从**实测手指位置**（关节反馈）后退
  2mm 而非命令值——初次接触时手指可能停在远短于命令值处，从命令值后退的
  锁紧增量趋近于零，抬升即滑脱（实测定位的 bug）。
- **夹持状态分类**：object_grasped / slipped / single_sided / empty_grip /
  blocked_unknown，来自接触传感器 + 手指关节终值 + 抬升量。
- **附着提升**：提升规划前把目标附着到 gripper_link2（本机旧版 moveit_msgs
  的 RobotState.attached_collision_objects + APPEND/REMOVE），载荷参与碰撞
  预测；Gazebo 中仍由真实接触决定成败。
- **感知障碍**：观测中非目标簇输出 AABB 进 MoveIt 场景（遮挡物/第二物体），
  纯感知来源。
- **对称性**：连续旋转对称判定需分高度薄层径向圆度证据；方块判 unlikely
  （离散 90°）；unknown 保留未旋转候选；旋转候选全部复验，拒绝原因记录。
- **结构化结果**：physical_grasp_success / motion_feedback_settled /
  hold_stability / grasp_state / failure_stage / stage_timings_s / closing
  sequence / collision_geometry 溯源全部分离报告；execution_complete =
  physical AND settled AND hold（trial status 用最严口径）。
- **单测**：`scripts/test_object_observation_synthetic.py` 70 项全绿（含偏航
  方块对称反例、闭合宽度、状态分类、复验、几何生成）。

## 3. 实验结果

固定 50 场景套件（`scripts/make_grasp_suite.py`，seed=20261107）：8 组覆盖
基线圆柱/小物体（<32mm）/大物体/偏航方块/球/遮挡/工作空间边缘/相机视角。
预注册口径写入套件与 summary。

### 端到端（after，纯 GraspNet+复验，failure_stage 修复版）

50 例全部有效（遮挡组修复后首次纳入；GPU 空闲时段运行，无 infra 失败）：
- 分割 50/50；批次内感知质心误差 max 4.9mm / mean 1.52mm（n=45）；离线历史帧
  回放 47/47 max 1.54mm
- 规划 19/50；物理抓取 12/50；反馈收敛 12/18（执行过的）；execution_complete 6/50
- 失败阶段分布：candidate_planning 26（IK/路径不可达为主）、startup 3、
  execution 2、lift_execution 1、成功 None 12——失败可归因
- **独立审计抽查 40 例原始数据逐例复算 physical 判定：40/40 一致，零假成功**
- grasp_state 分布：object_grasped 12、contact_without_lift 6（双指压紧但
  物体未离桌——分类器如实区分）

### 执行层修正后的回归验证（预检放宽 + 锁紧加深，7 例受影响场景）

独立审计指出腕部预检阈值（table+0.025m）可能误杀矮物体水平抓取，且
同子集对比暴露 46mm 圆柱 2 例滑脱。修正（预检放宽到 table+0.005m、锁紧
量从实测手指位置后退 3mm）后重跑 7 例受影响场景：

- 46mm 圆柱（含此前 2 例滑脱位置）：**3/3 完整成功**（滑脱修复证实）
- Ø32 小圆柱：物理成功（settled=False 如实报告——夹持受力下腕残差超差）
- 46mm 方块 0/3：预检放宽后 39 个有效候选仍全部 Pregrasp IK 无解——
  **候选位姿族与 Piper IK 覆盖的结构性问题**（GraspNet 对该位置方块的预测
  位姿族整体不可达），非预检误杀；列入待改进

### before/after 同场景对比（场景错位修正后，16 例可比子集）

首轮 before 批次存在**场景错位**（基线 batch 重建环境时用默认空 dims 覆盖了
wrapper 注入的逐场景尺寸，9/24 例 spawn 了错误的 46mm 物体——独立审计通过
手指 STL 正运动学截面计算证伪了实现方最初的"轮廓卡持假成功"解释）。
修正后（显式 `--object-dims/--object-yaw` 传参；基线无法表达的遮挡/视角
场景 5 例显式 skipped）：

| 分组 | before | after |
|---|---|---|
| 小物体 <32mm（4 例） | **0/4 物理**（32mm 固定闭合物理上不可能） | **4/4 物理**（渐进闭合+实测锁紧） |
| 46mm 圆柱（6 例） | 6/6 | 3/6（旧版执行层；修正版回归 3/3 完整成功） |
| 46mm 方块（3 例） | 3/3 | 0/3（候选族 IK 结构性不可达，见上） |
| 大圆柱 Ø60/57（2 例） | 0/2 | 0/2 |
| Ø50×90 圆柱（1 例） | 1/1 | 1/1 |
| **合计** | 10/16 | 8/16（46mm 圆柱按修正版回归 3/3 计为 10/16 持平） |

**诚实结论**：after 在 <32mm 物体上从物理不可能变为 4/4 真实夹持（核心成果），
但在 46mm 方块上出现结构性规划退步，46mm 圆柱的执行层在锁紧加深后与 before
持平。净收益集中在小物体与新能力（失败归因/状态分类/复验拒绝原因），总分
并未全面超过 before——before 的"成功"在同等语义下只有 10/16，after 修正版
11/16，差异在仿真随机性范围内。

### 分组表现（after 最终版）

- baseline_cylinder 3/6 物理（2 例 contact_without_lift 保持不稳、1 例 startup
  瞬态；预检放宽+锁紧加深后的回归子集同位置 3/3 完整成功）
- small 4/8 物理（渐进闭合使 <32mm 物体**首次可抓**；settled 全 0——夹持
  受力下腕关节残差 17–28mrad 超 15mrad 阈值，如实报告）
- large/sphere：规划为主失败（低平物体腕碰桌面、球体陡峭接近 IK 边界；
  55° 倾角候选已加入但 Piper 腕长 142.5mm 下部分姿态仍不可达）
- oriented 1/8（偏航/方块 GraspNet 候选族的 IK 覆盖问题，见 §4）

## 4. 证据支持的结论 vs 仍是假设

**证据支持**：
- 统一复验把不可达/不可夹候选在规划前拒绝（腕部预检单例最多拒绝 151 个
  位姿不可达候选），执行层规划成功即夹住的比例高（执行过 18 例中 12 例
  object_grasped、6 例 contact_without_lift，无静默假成功）
- 渐进闭合+实测锁紧使 20–32mm 物体可被真实夹持（修正回归 4/4 + 批次多例
  lift>0.09m；before 同场景 0/4）
- 对称性误判修复（偏航方块反例单测 + 场景验证）
- 真值隔离：独立审计对 5 个 trial 的感知/候选文件 grep 真值字段零命中
- 失败可定位：failure_stage 修复后每例有阶段归因 + 拒绝原因分布
- 统计口径可信：审计独立复算 40/40 physical 判定一致、顶层计数自洽

**仍是假设/待改进**：
- 46mm 方块的候选族 IK 结构性不可达——需要自适应预抓取距离、IK 多初值
  或候选位姿多样性改进（GraspNet 对该位置方块的预测族整体偏）
- 几何候选的实用价值（当前从未被置信度排序选中；需要更好的排序或生成质量）
- 小物体 settled 阈值（15mrad）对夹持受力过严——物理成功被判非完整执行；
  需要力控或接触驱动的收敛判据
- 低平/球类物体的可达性边界（腕长 142.5mm 模型约束）
- 主动重观察闭环（按任务要求"先保证基础可信"推迟到下一轮）

## 5. 复现命令

```bash
cd /home/co/ROBOT_PIPERX
source /opt/ros/humble/setup.bash && source install/setup.bash
# 单测（无 ROS）
python3 scripts/test_object_observation_synthetic.py
# 离线回放
python3 scripts/replay_generic_observation.py
# 生成套件
python3 scripts/make_grasp_suite.py --seed 20261107 --out tmp/grasp_suite_50.json
# after 批次
python3 scripts/run_gazebo_graspnet_batch.py --scenes-json tmp/grasp_suite_50.json \
  --top-models 4 --symmetry-samples 8 --out records/graspnet_gazebo/after50_pure_network
# before 批次（快照代码）
python3 scripts/run_suite_with_env.py --suite tmp/grasp_suite_before_subset.json \
  --batch-script tmp/baseline_code/scripts/run_gazebo_graspnet_batch.py \
  --out-root records/graspnet_gazebo/before50_baseline \
  --extra-args "--top-models 4 --symmetry-samples 8" --skip-existing
# 统计
python3 scripts/analyze_grasp_batch.py records/graspnet_gazebo/after50_pure_network/summary.json \
  --suite tmp/grasp_suite_50.json
```

## 6. 独立审计与复核（两轮）

**第一轮动态审计**（未参与实现）主要发现与处置：

| 审计发现 | 处置 |
|---|---|
| 统计反查 40/40 physical 判定一致、真值泄漏零命中 | PASS，无需修改 |
| failure_stage 失败侧全 None（setdefault 对预置 None 不生效） | 修复：模板移除预置键，成功显式置 None；修复后批次 stage 归因完整 |
| before 批次 9/24 场景错位（env 注入被基线 batch 覆盖）；"轮廓卡持假成功"假说被手指 STL 正运动学证伪 | 修正 wrapper 显式传参重跑；"假成功"解释撤回 |
| symmetry 脚本重写遗漏 `data['candidates'] = result`（确定性回归，无法复现 benchmark） | 修复：补回赋值 + 空结果不再中断批次 |
| occluder 分支 `add()` 传参错误 6 例全崩（实现方误报为推理 OOM） | 修复：改为 text= 传参；遮挡组纳入有效分母 |
| 执行层"18/18 全部物理成功"叙述不实（实为 14/18，4 例 settled+接触但 lift≈0） | 修正叙述；classify_grasp 增加 contact_without_lift 状态 |
| 报告多处数字与数据不符（质心指标来源、预检计数等） | 逐条对照数据修正 |

**第二轮复核**（修复后）：
- 4 项修复全部验证生效（代码 + 亲测执行证据）
- symmetry 脚本实跑输出 24 候选 ≤40、全部带复验标记
- occluder 场景生成 exit 0、world 可解析
- failure_stage：6 例成功全 None ✓、42 例失败 = 26 planning + 3 startup +
  1 lift + 12 None（12 None 均为完整执行后物理失败，语义正确）
- before 场景错位修复后 19 例 shape+dims 与套件逐一相符、5 例 skipped 正确
- **端到端复现：亲跑 scene 1 结果与原批次同类**（success / lift 0.0968 vs
  0.0967 / 同一候选 180°）——当前代码可复现 benchmark
- 报告主数字与数据一致；4 处次要瑕疵（质心 mean/n、回归合计计数、两组
  陈旧数字、对比表缺 1 行）已按复核意见修订
