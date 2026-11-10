# Piper-X 任意形状抓取开发交接

更新时间：2026-11-08

> **最新状态**：2026-11-08 完成一轮大规模重构与两轮独立审计（候选统一复验、
> 渐进闭合+实测锁紧、对称性圆度证据、failure_stage 归因、真值隔离强化、
> 50 场景固定套件与 before/after 对比）。结论、数据与审计记录见
> `docs/REFACTOR_REPORT_20261108.md`——该文档优先于本文档 §5/§9 的历史数字。

项目根目录：`/home/co/ROBOT_PIPERX`

## 1. 交接目标

当前系统已经从“固定尺寸圆柱的几何抓取”推进到“GraspNet RGB-D 候选 + Piper 仿真抓取”，并验证了固定尺寸圆柱和方块。下一阶段目标是：

1. 去掉固定物体形状、尺寸和直立状态先验。
2. 从 RGB-D/点云自动得到目标区域、包围盒/凸包和碰撞几何。
3. 将 GraspNet 的通用候选经过 Piper 夹爪、碰撞、逆解和完整路径筛选。
4. 在 Gazebo 中测试不同尺寸、不同形状、不同旋转和部分遮挡。
5. 仿真稳定后，再设计实机校准与低速验证。

不要直接声称已经支持“任意物体”。只有经过多形状、多尺寸、多姿态测试后，才能更新这一结论。

## 2. 硬件与软件

- 机械臂：AgileX Piper，原厂平行夹爪。
- 相机：腕部上侧安装的 Okulo RGB-D；相机随腕部旋转，不随夹爪开合旋转。
- 仿真：Gazebo 11、ROS 2 Humble、MoveIt 2。
- GPU：RTX 5060 Laptop，约 8 GB 显存。
- 相机实机点云需要通过 Okulo SDK 获取；相机存在畸变，不能直接把深度图当普通针孔相机重新反投影。
- GraspNet Baseline 使用独立 `.graspnet-venv`，通过 `.pth` 引用 `.isaac-venv` 中的 PyTorch 2.7.0 + CUDA 12.8。
- CUDA 编译组件位于 `third_party/graspnet-cuda-12.8/`，按 `sm_120` 编译了 pointnet2/knn。

注意：`/home/co/ROBOT_PIPERX` 是当前机器人项目。`/home/co/文档/ChatGPT/ROBOT` 是另一个目录，主要放相机 SDK 等资料。不要混用。

## 3. 安全与操作边界

- 所有现有自动抓取测试只在 `ROS_DOMAIN_ID=183`、`ROS_LOCALHOST_ONLY=1` 和独立 Gazebo master 中运行。
- 任何视觉/规划试验默认不得启动 CAN 驱动，也不得向实机发送控制命令。
- 不要用 Gazebo 真值替代感知输入。位置真值只能用于场景生成和最终评估。
- 不要通过固定连接、瞬移或直接附着物体制造成功结果。
- 判断抓取结果时必须区分：
  - 物理抓取成功：双指接触且物体实际被抬起并保持。
  - 轨迹反馈收敛：抬升终点各关节最终误差在严格容差内。
- 末端反馈超时不能直接覆盖已经发生的物理抓取成功；反过来，末端收敛也不能证明物体被夹稳。

## 4. 当前流水线

主要路径（generic 模式，2026-11-07 起）：

```text
Gazebo RGB-D
  -> 图像时刻 TF + 图像时刻关节状态同步（取与图像时间戳最近的关节消息）
  -> 彩色目标掩码和掩码内点云
  -> 通用观测：桌面直方图估计 -> ROI -> 欧式聚类 -> 统计滤波
     -> 支撑面补全凸包 + 重力对齐 OBB + 尺寸 + 对称性假设
  -> GraspNet Baseline 候选
  -> GraspNet 坐标到 Piper TCP 转换（共享筛选：开口 + 夹爪体积 + 指尖邻域）
  -> 按对称性假设扩展姿态（likely 连续 / unlikely 离散 / unknown 只保留原始候选）
  -> MoveIt 凸包 mesh 碰撞体 + 预抓取、接近、目标避碰
  -> 仿真闭合、保持、抬升和结构化结果评估
```

关键文件：

| 路径 | 作用 |
| --- | --- |
| `scripts/start_gazebo_grasp.sh` | 启动独立 ROS 域和 Gazebo 场景，负责 MoveIt、相机桥和控制器 |
| `sim/gazebo/grasp/capture_graspnet.py` | 采集同一时刻的 RGB-D、TF、关节状态；`--observation-mode generic` 为默认，cylinder/cube 是 legacy 回归分支 |
| `src/piperx_lab/piperx_lab/object_observation.py` | 通用观测共享结构 `GenericObjectObservation` 与无 ROS 分割流水线 |
| `src/piperx_lab/piperx_lab/collision_geometry.py` | 凸包质量门槛与 MoveIt mesh 碰撞体 |
| `src/piperx_lab/piperx_lab/grasp_candidate_filter.py` | 共享候选筛选（开口、夹爪体积、指尖点邻域） |
| `scripts/graspnet_offline_demo.py` | 运行官方 GraspNet Baseline 推理并保存候选 |
| `scripts/prepare_graspnet_sim_candidates.py` | 将 GraspNet 候选转换成 Piper TCP 候选 |
| `src/piperx_lab/piperx_lab/graspnet_adapter.py` | 开口、目标点关联、预抓取距离和坐标轴转换 |
| `scripts/augment_cylinder_grasp_symmetry.py` | 绕观测竖直轴扩展候选姿态，generic 观测按对称性假设驱动 |
| `sim/gazebo/grasp/run_grasp.py` | MoveIt 路径检查、凸包碰撞体、夹爪闭合、物理抬升和结构化结果 |
| `scripts/run_gazebo_graspnet_batch.py` | 端到端批次，支持参数化形状/尺寸/偏航与结构化 summary |
| `sim/gazebo/grasp/make_scene.py` | 场景生成：cylinder/cube/sphere/box/bar/mesh + 随机尺寸/偏航/质量/摩擦 |
| `scripts/test_object_observation_synthetic.py` | 通用观测合成点云单测（无 ROS） |
| `scripts/replay_generic_observation.py` | 历史 frame.npz 离线回放评分 |
| `docs/GENERIC_OBJECT_OBSERVATION.md` | 通用观测与自动碰撞几何说明 |
| `docs/GRASPNET_BASELINE.md` | GraspNet 实验说明和结果 |
| `docs/GAZEBO_VISION_GRASP.md` | 早期视觉圆柱抓取和批次流程 |
| `docs/GAZEBO_CYLINDER_GRASP.md` | 早期已知位置抓取、摩擦和 PID 试验 |

## 5. 当前支持和未支持范围

已测试（2026-11-07，generic 观测 + 凸包碰撞）：

- 红色直立圆柱，直径 46 mm、高 80 mm，质量 60 g（含随机尺寸圆柱 27–57 mm、高 53–90 mm 试点）。
- 红色直立方块，边长 46 mm，质量 60 g。
- 球、参数化 box/bar、STL/OBJ mesh 场景（make_scene 参数化）。
- 随机 XY 位置，有限区域为 `x=0.40–0.50 m`、`y=-0.05–0.05 m`。
- 随机偏航（box 试点 ±180°）。
- 固定上方腕部视角和理想仿真深度。
- 通用观测：颜色掩码 + 桌面直方图 + 聚类 + 统计滤波 + 支撑面补全凸包 + OBB。
- 自动碰撞几何：观测凸包 -> MoveIt mesh 碰撞体，低质量点云显式拒绝。
- 方块 GraspNet 候选，以及模型候选全部不可达后的几何上抓回退。

尚未支持：

- 任意材质（透明、反光、无纹理、强阴影）。
- 横放、翻滚、堆叠和杂乱遮挡。
- 从完整点云自动估计任意物体 6D 位姿（当前 OBB/对称性为启发式）。
- 实机低速抓取、夹爪滑移和手眼漂移验证。
- 抓住物体后的完整携物碰撞规划。当前抬升预规划只验证空手手臂姿态。
- 复杂背景下的颜色掩码（第一版分割仍依赖红色目标）。

Piper 当前仿真的打开宽度为 70 mm。任何观测物体都需要检查夹爪开口、指尖几何、接近方向和桌面碰撞。

## 6. GraspNet 环境和来源

- 官方源码：<https://github.com/graspnet/graspnet-baseline>
- 本机源码：`third_party/graspnet-baseline/`
- 本机源码版本：`280c215129f759ed8649cb4e89fc5dfee55f4f80`
- 权重：`third_party/graspnet-baseline/checkpoints/checkpoint-rs.tar`
- 权重 SHA-256：`60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868`
- 官方 Google Drive 当时受配额限制；本次权重来自第三方发布页：<https://huggingface.co/dgrachev/a2_pretrained>
- 官方未提供用于逐字节核对的校验值，因此不能证明第三方权重与官方文件完全相同。
- 权重以 `weights_only=True` 加载并严格校验网络参数结构。
- 官方项目说明其代码、数据和模型供非商业使用；其他用途必须遵守原项目许可。

上游代码中 `torch.cross` 没有指定维度，当候选数恰好为 3 时可能产生错误旋转矩阵。当前源码已应用：

```text
patches/graspnet_baseline_cross.patch
```

重新检出 GraspNet 源码后，需要重新应用该补丁。

AnyGrasp 尚未接入。其 SDK 需要机器特征码并申请许可证；目前没有许可证和授权 SDK 库。

## 7. 当前相机目标观测

`capture_graspnet.py` 默认运行通用观测（`--observation-mode generic`），细节见
`docs/GENERIC_OBJECT_OBSERVATION.md`。要点：

- 颜色掩码（第一版）给出目标表面点云；桌面高度来自全场景 z 直方图主峰（回退需显式先验）。
- ROI 内全场景点与掩码点一起欧式聚类，选与掩码重叠最大的簇（自然包含白色顶盖）。
- 统计滤波去离群点；凸包经支撑面阴影补全闭合；OBB 重力对齐。
- `dimensions_m` = [水平 OBB 两轴, 支撑面到顶点高度]；质心为补全凸包体心。
- 输出 `GenericObjectObservation` JSON + object_points.npz + object_hull.ply + mask.png。
- 分割/质量不达标显式拒绝，不用固定尺寸兜底。
- 关节同步取与图像时间戳最接近的关节消息（近 2 s 历史），对处理延迟不敏感。

legacy 圆柱/方块分支（`--observation-mode cylinder|cube`）原样保留作回归：
- 圆柱：红色点云拟合圆柱，直径必须接近 46 mm，轴向必须接近竖直。
- 方块：红色点云计算 XY 范围，边长固定 46 mm。

目标观测和 GraspNet 使用同一帧，但 GraspNet 推理本身不读取 Gazebo 真值。

## 8. 抓取候选与执行

GraspNet 输出 Nx17 候选项。当前转换假设 GraspNet 的：

- 局部 `+X` 是接近轴。
- 局部 `+Y` 是夹爪开合轴。
- 指尖参考点位于 `x=depth`。

Piper 默认夹爪和 GraspNet 理想夹爪不同，因此这是近似几何对齐，不是充分接触证明。模型分数不是抓取成功概率。

候选级筛选在共享模块 `piperx_lab.grasp_candidate_filter`：开口 ≤ 70 mm、
夹爪体积目标关联（`points_in_jaw_volume`）、以及以**指尖中点 `T_base_tip`**（夹持中心）
到观测中心的目标邻域检查。注意法兰 `T_base_tcp` 在指尖后方 142.5 mm，
对邻域检查而言天然远离物体，不能用。

对称姿态扩展由观测的对称性假设驱动：`likely` 均匀采样、`unlikely` 按记录的离散阶
（方块 90°、长条 180°）、`unknown` 只保留未旋转原始候选（`--force` 才强制旋转）。
旋转后的候选必须保留：

```json
{
  "rotated_target_contact_revalidated": false,
  "fallback": true
}
```

方块专用的 `--geometry-fallback` 只能在 GraspNet 候选全部失败后启用（generic 观测下
使用观测尺寸而不是 0.046），结果必须标记为几何回退，不能冒充 GraspNet 成功。

`run_grasp.py` 用观测凸包生成 MoveIt mesh 碰撞体并记录溯源；凸包尺寸越界、
不闭合、低于支撑面 5 mm 以上或与观测 AABB 不符时直接拒绝抓取。保持检查模式：
legacy 圆柱=倾角+位移，legacy 方块=不检查，generic 按对称性假设选
`upright_tilt`（likely，倾角 15° + 位移 3 mm）或 `displacement_only`（位移 5 mm）。
结果 JSON 含结构化字段：`physical_grasp_success`（双指接触+抬升>0.07 m+XY<0.06 m，
不含保持门槛）、`hold_stability`、`failure_stage`、`failure_reason`、
`object_dimensions_m`、`object_pose`、`collision_geometry`。

## 9. 当前结果及统计口径

原始历史结果里有 `success=false`，可能是因为抬升轨迹已经发生、物体已经抬起，但末端关节反馈没有完全进入严格容差。统计时必须读取 `object_samples.lifting`、`initial_object_xyz` 和 `finger_contacts` 重新判断物理抓取。generic 批次 summary 现在直接给出 `physical_grasp_success` 与 `success` 两个口径。

### generic 观测回归（2026-11-07）

- 离线回放 47 个历史 trial frame（`scripts/replay_generic_observation.py`）：
  - 分割成功 47/47，质心误差 max 1.54 mm / mean 1.04 mm（legacy 对照 max 1.92 / mean 1.16）。
  - 尺寸误差最大 4.3 mm（47/47 满足 <5 mm 门槛）。
- 随机十位置圆柱 `seed=20261106`，generic 观测 + 凸包碰撞
  （`records/graspnet_gazebo/generic_random10_seed20261106/`）：
  - 分割 10/10，碰撞几何 10/10，规划 9/10。
  - 物理抓取成功：9/10（历史 legacy 同种子：8/10）。
  - 严格 success（含保持检查）：9/10；末端反馈收敛 6/10。
  - 唯一失败为 t1，与历史失败位置相同（工作空间右缘，32 个候选全部预抓取 IK 无解）。
  - 注意：该批运行于凸包支撑面补全之前；补全后需按 §12 命令重跑确认。
- 方块试点 `seed=403`，generic 观测 + 几何回退
  （`records/graspnet_gazebo/generic_cube_seed403/`）：物理抓取 3/3（历史 legacy：3/3）。
- 合成单测：37 项全绿（46 mm 圆柱/46 mm 方块/40 mm 球/随机 box 重建误差 0.0 mm）。

### 新形状试点（分段：分割与几何测量已验证，抓取统计待 GPU 恢复后重跑）

球/随机尺寸 box/随机尺寸圆柱试点中，通用观测对全部新形状给出正确分割与尺寸
（如球 [36,32,40] mm、随机圆柱 [48,49,90]/[57,57,53]/[26,27,75] mm、随机 box 含
±180° 偏航），但抓取统计尚未完成：首轮暴露 yaw 采样越界（已修）、unknown 对称性
失败（已改为保留原始候选）、单模型几何复制与陡峭接近角 IK 无解两类规划失败
（试点改用 `--top-models 4`）。GPU 驱动在 2026-11-07 上午进入 CUDA unknown error
状态（nvidia-smi 正常、CUDA 上下文创建失败），重跑被阻塞。

### 早期 legacy 结果（历史基线，口径不变）

- 早期规则几何 `random_seed42` 十例：物理 10/10；严格末端反馈收敛 8/10。
- 同一组十个位置改用 GraspNet：物理 9/10；严格收敛 6/10；唯一未抓起为第 5 例
  （记录：`records/graspnet_gazebo/geometry_seed42_retested_with_graspnet/`）。
- GraspNet 随机十位置 `seed=20261106`：物理 8/10；严格收敛 5/10；2/10 无候选通过
  完整路径（记录：`records/graspnet_gazebo/random10_seed20261106/`）。
- 七个历史失败位置复测：物理 6/7
  （记录：`records/graspnet_gazebo/retest_prior_failures_valid/`）。
- 三个方块几何回退试点：物理 3/3；严格收敛 2/3
  （记录：`records/graspnet_gazebo/cube_pilot_seed403/`）。

现有仿真成功判据的历史版本是：

- 闭合和抬升阶段双指都有接触。
- 最终抬升超过 70 mm。
- 水平偏移小于 60 mm。
- 圆柱最后 10 个样本倾角小于 15°，相对首样本位移小于 3 mm。

generic 观测下保持检查按对称性假设选择（见 §8）；这些阈值仍属早期仿真口径，
不适合直接作为任意物体最终指标。

## 10. 任务状态（2026-11-07 更新）

### P0：通用对象观测接口 —— 已完成

`piperx_lab.object_observation.GenericObjectObservation` 已由真实 C1W 和 Gazebo 共用
（结构见 `docs/GENERIC_OBJECT_OBSERVATION.md` §3）。颜色掩码第一版、图像/内参反投影
（实机应优先 SDK XYZ）、图像时刻 TF + 关节状态、桌面直方图估计、欧式聚类、
统计滤波、凸包/OBB/尺寸/对称性假设、掩码/NPZ/PLY/JSON 输出均已实现；
合成单测 `scripts/test_object_observation_synthetic.py` 与历史帧回放
`scripts/replay_generic_observation.py` 覆盖。

### P0：自动碰撞几何 —— 已完成

`run_grasp.py` 读取观测凸包 PLY 生成 MoveIt mesh `CollisionObject`（桌面单独），
记录溯源（顶点数/三角数/闭合性/时间戳/滤波标记），低质量点云显式拒绝，
不再使用硬编码尺寸兜底。

### P1：通用候选筛选 —— 已完成（候选级）

共享模块 `piperx_lab.grasp_candidate_filter`：开口、夹爪体积关联、指尖点目标邻域。
MoveIt 阶段性验证（预抓取 IK、路径、最后 10 mm 接触、闭合/抬升）留在
`run_grasp.py`，本就与物体形状无关。候选来源与评分保留，generic 与 legacy 统计分开记录。

### P1：仿真对象扩展 —— 已完成

`make_scene.py` 支持 `cylinder|cube|sphere|box|bar|mesh` + `PIPER_SIM_OBJECT_DIMS`
（或批次 `--object-dims auto`）、`PIPER_SIM_OBJECT_YAW_DEG`、`PIPER_SIM_OBJECT_FRICTION`、
`PIPER_SIM_OBJECT_MASS`、`PIPER_SIM_OBJECT_MESH`。模型名仍叫 `cylinder`（接触、
状态查询和评估逻辑依赖）。胶囊体未实现（SDF 1.6 无原生胶囊）。横放、倾斜、
堆叠和遮挡尚未加入。

### P1：批次结果结构 —— 已完成

summary/trial 含 `segmentation_success`、`collision_geometry_success`、
`candidate_count`、`planning_success`、`physical_grasp_success`、
`motion_feedback_settled`、`max_object_lift_m`、`hold_stability`、`failure_stage`、
`failure_reason`、`object_shape`、`object_dimensions_m`（感知）、
`object_dimensions_scene_m`（场景真值）、`object_pose`。`success` 仍保留但含义
（含保持门槛）已在 `failure_stage`/`hold_stability` 下拆解。

### 后续任务

1. 按失败样本改进候选生成：新形状试点中"单模型几何复制成全部候选"和
   "陡峭接近角预抓取 IK 无解"是主要失败模式（可试更多 ROI/模型、自适应预抓取距离）。
2. 横放、倾斜、堆叠和遮挡场景。
3. 实机 C1W 接入同一观测接口（SDK XYZ 路径）。
4. 需要时申请 AnyGrasp 许可或收集数据微调 GraspNet。
5. VLA 作为高层策略复用同一观测/抓取/规划接口。

## 11. 建议的验收门槛

第一轮通用基本形状仿真可先采用：

- 分割成功率大于 95%。
- 目标点云与真实位置的质心误差小于 5 mm。
- 自动碰撞体尺寸误差小于 5 mm。
- 抓取候选规划成功率大于 80%。
- 物理抓取成功率大于 80%。
- 抓取后保持 3 秒，相对初始抓取点位移小于 5 mm。
- 每一项同时记录严格末端反馈是否收敛，但不能用它替代物理成功。

完成固定直立基本形状后，再测试随机偏航、横放、部分遮挡和多物体场景。

## 12. 复现命令

运行前：

```bash
cd /home/co/ROBOT_PIPERX
source /opt/ros/humble/setup.bash
source install/setup.bash
```

离线验证（无 ROS、无 Gazebo）：

```bash
python3 scripts/test_object_observation_synthetic.py     # 合成单测
python3 scripts/replay_generic_observation.py            # 历史 frame 回放评分
```

随机十位置圆柱（generic 观测）：

```bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --trials 10 --seed 20261106 \
  --out records/graspnet_gazebo/new_random10
```

legacy 固定形状观测回归（旧行为对照）：

```bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --observation-mode cylinder --trials 10 --seed 20261106 \
  --out records/graspnet_gazebo/new_random10_legacy
```

指定位置复测：

```bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --xy 0.48921795677048452,-0.04130611673705839 \
  --xy 0.48094304566778268,-0.049350124032193902 \
  --out records/graspnet_gazebo/new_fixed_xy
```

方块试点：

```bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape cube --trials 3 --seed 403 \
  --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/new_cube
```

新形状试点（球 / 随机尺寸 box / 随机尺寸圆柱）：

```bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape sphere --trials 3 --seed 911 \
  --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/new_sphere
python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape box --object-dims auto --object-yaw auto \
  --trials 3 --seed 912 --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/new_randombox
python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape cylinder --object-dims auto \
  --trials 3 --seed 913 --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/new_randomcyl
```

图形界面观察并让最后一个场景保持打开：

```bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --gui --keep-open-last \
  --xy 0.48921795677048452,-0.04130611673705839 \
  --out records/graspnet_gazebo/gui_case
```

## 13. 代码和仓库注意事项

- 当前 `/home/co/ROBOT_PIPERX` 检查时没有暴露 `.git` 目录；进行提交前先确认真实仓库和分支，不要假设可以 `git status` 或 push。
- `.gitignore` 已忽略 `.graspnet-venv/`、`.isaac-*/`、`third_party/*`、`records/*` 和构建产物。
- 不要提交权重、许可证文件、虚拟环境、CUDA 包或大体积实验记录，除非目标仓库有明确的版本管理要求。
- 保留用户已有改动；不要执行 `git reset --hard`、`git checkout --` 或清理整个工作区。
- `start_gazebo_grasp.sh` 的终止依赖进程组。父 shell 被中断后，`gzserver`、`ros2 launch move_group` 或 `robot_state_publisher` 可能成为孤儿进程。重新启动前用精确 PID/进程组清理，不要误杀实机驱动。

## 14. 推荐执行顺序

1. ~~先实现通用对象观测接口，并保留圆柱/方块回归路径。~~（2026-11-07 完成）
2. ~~用凸包生成自动碰撞体，替换硬编码尺寸。~~（完成）
3. ~~在同一固定尺寸方块、圆柱上确认规划成功率不下降。~~（完成：圆柱 9/10、方块 3/3）
4. ~~加入随机尺寸的方块、圆柱和球。~~（场景与批次支持完成；试点统计见 §9）
5. ~~加入不规则 STL/OBJ 和随机偏航。~~（场景支持完成；mesh 抓取试点待做）
6. 加入横放、堆叠和遮挡。
7. 依据失败样本改进 GraspNet 到 Piper 的坐标/接触映射与候选生成。
8. 需要时申请 AnyGrasp 许可或收集数据微调 GraspNet。
9. 仿真达到验收门槛后，再设计实机低速验证。

优先解决“任意基本几何形状和尺寸”，再进入复杂材质、杂乱场景和 VLA。VLA 适合作为后续高层任务策略，应复用同一个观测、抓取和规划接口。
