# 通用对象观测与自动碰撞几何

更新时间：2026-11-07

本文描述从"固定形状先验拟合"切换到"形状无关观测"后的流水线。历史背景见
`docs/GRASPNET_BASELINE.md`；交接总览见 `docs/AGENT_HANDOFF_ARBITRARY_OBJECT_GRASP.md`。

## 1. 目标与边界

- 观测不再依赖固定直径/边长/直立先验；桌面高度由场景点云估计。
- 颜色掩码仍是第一版分割输入；开放词分割尚未引入。
- 只观测可见表面：凸包缺底面；支撑高度假设物体静置于支撑面上。
- 观测失败必须显式拒绝，不得回退到硬编码尺寸。
- 真值仍只用于场景生成与最终评估，从不进入观测。

## 2. 流水线

```text
Gazebo RGB-D（同一帧 color/depth/info + 图像时刻 TF + 关节状态）
  -> 颜色掩码 + 掩码内相机系点云           （capture 回调内，轻量）
  -> 全帧 XYZ 组织点云（frame.npz 原有输出）
  -> [main 中] 桌面 z 直方图主峰 + 中值精化   （失败回退显式先验并记录来源）
  -> 掩码邻域 ROI（xy 半径 + 支撑面上方高度窗）
  -> 全场景 + 掩码点欧式聚类（cKDTree + 并查集，4 mm）
  -> 选择与掩码重叠最大的簇；统计滤波（kNN 均距）
  -> ConvexHull 凸包 + 重力对齐 OBB（水平轴来自 PCA 投影）
  -> 凸包支撑面补全：点云投影到支撑面（阴影）并入凸包，闭合底面；
     近平面簇由 hull/OBB 体积比门槛（默认 0.05）拒绝
  -> dimensions_m = [水平 OBB 两轴, 支撑面到顶点高度]
  -> 质心 = 补全后凸包体心
  -> 对称性假设（likely / unlikely / unknown，仅启发式）
  -> 保存 mask.png / frame_object_points.npz / frame_object_hull.ply + JSON
```

实现要点：

| 路径 | 作用 |
| --- | --- |
| `src/piperx_lab/piperx_lab/object_observation.py` | 共享观测结构 `GenericObjectObservation`、分割流水线、PLY 读写；无 ROS 依赖，可离线测试 |
| `src/piperx_lab/piperx_lab/collision_geometry.py` | 凸包质量门槛（尺寸/闭合性/支撑面/与观测 AABB 一致）与 MoveIt mesh CollisionObject |
| `src/piperx_lab/piperx_lab/grasp_candidate_filter.py` | 共享候选筛选：开口、夹爪体积关联、指尖点目标邻域 |
| `sim/gazebo/grasp/capture_graspnet.py` | `--observation-mode generic|cylinder|cube`；generic 为默认，legacy 两分支保留作回归 |
| `sim/gazebo/grasp/run_grasp.py` | 凸包 mesh 碰撞体；结构化结果字段；保持检查泛化 |
| `scripts/test_object_observation_synthetic.py` | 合成点云单测（无 ROS） |
| `scripts/replay_generic_observation.py` | 历史 frame.npz 离线回放评分 |

## 3. 数据结构

`GenericObjectObservation`（`frame_observation.json`，source=`rgbd_object_segmentation`）：

- `stamp` / `base_frame` / `camera_frame` / `joint_positions_rad` / `T_base_camera`
- `centroid_base_m`：可见面凸包体心（偏高，已知偏置）
- `obb_center_base_m`：支撑盒中心，run_grasp 的规划中心取此值
- `dimensions_m`：[水平 OBB 长轴, 短轴, 支撑面到顶点高度]
- `obb_axes_base`：重力对齐 OBB 三轴（行向量）
- `mask_path` / `points_base_path` / `convex_hull_path`：相对观测 JSON 的文件
- `segmentation_confidence`：掩码重叠率 × 统计滤波内点率的启发式，非概率
- `quality`：点数、桌面估计来源与峰值、凸包/OBB 体积比、PCA 竖直倾角、AABB、ROI 裁剪标记
- `symmetry.vertical_axis`：`likely`（连续旋转对称假设）/ `unlikely`（记录离散阶 [0,90,180,270] 或 [0,180]）/ `unknown`；仅启发式，不是形状分类。
  对称扩展脚本对 `unknown` 不再失败，而是只保留未旋转的原始候选（`--force` 才强制均匀旋转）。

## 4. 执行侧变化

- `run_grasp.py` 读取观测中的凸包 PLY 生成 MoveIt mesh 碰撞体，并在报告中记录
  `collision_geometry` 溯源（顶点数/三角数/闭合性/时间戳/滤波标记）。
- 凸包门槛失败（尺寸越界、不闭合、低于支撑面 5 mm 以上、与观测 AABB 不符）直接
  拒绝抓取；不再使用硬编码尺寸兜底。
- 保持检查模式：legacy 圆柱 = 倾角+位移（不变）；legacy 方块 = 不检查（不变）；
  generic 观测按对称性假设选 `upright_tilt`（likely）或 `displacement_only`（5 mm）。
- 结果 JSON 新增结构化字段：`physical_grasp_success`（双指两阶段接触 + 抬升 > 0.07 m
  + XY < 0.06 m，不含保持门槛）、`hold_stability`、`failure_stage`、`failure_reason`、
  `object_dimensions_m`、`object_pose`、`collision_geometry`。
- 候选邻域检查使用指尖中点 `T_base_tip`（夹持中心），不是法兰 `T_base_tcp`
  （法兰在指尖后方 142.5 mm，远离物体是设计使然）。

## 5. 场景参数化（P1）

`make_scene.py` 通过环境变量支持：

- `PIPER_SIM_OBJECT_SHAPE`：`cylinder|cube|sphere|box|bar|mesh`
- `PIPER_SIM_OBJECT_DIMS`：逗号尺寸（圆柱 `直径,高`；box/cube/bar `dx,dy,dz`；球 `直径`）
- `PIPER_SIM_OBJECT_YAW_DEG`、`PIPER_SIM_OBJECT_FRICTION`、`PIPER_SIM_OBJECT_MASS`
- `PIPER_SIM_OBJECT_MESH`：STL/OBJ 路径（惯性用 AABB 盒近似，已记录）

Gazebo 模型名保持 `cylinder`、link `body`、碰撞 `cylinder_collision`——接触检测、
`GetEntityState` 与评估逻辑依赖该名称，改名需同步全部订阅方。

批次新增：`--object-dims auto`（每例试随机可抓取尺寸：水平 ≤ 60 mm 适配 70 mm 开口）、
`--object-yaw auto`（每例试 ±180° 随机）、`--object-mesh`。新形状试点建议
`--top-models 4`：对称旋转只覆盖最好的 1~2 个模型抓取时，一种坏的接近几何
就会复制成全部候选。

## 6. 已知限制

- 只测过直立单物体；横放、堆叠、遮挡未测。
- 凸包底面来自支撑面阴影补全（显式假设物体静置于支撑面上），不是观测。
- OBB 水平尺寸来自 PCA 轴投影，偏航方块可能比真实边长偏大最多 ~5 mm（凸包本身不受影响）。
- `segmentation_confidence` 与对称性分类是启发式。
- mesh 形状惯性为 AABB 近似；胶囊体未实现（SDF 1.6 无原生胶囊）。
- 相机桥仍是理想深度，未模拟 Okulo 的 ToF 噪声与畸变。

## 7. 图像时刻关节同步

早期实现把"最新收到的 /joint_states"与图像时间戳比对；一旦回调处理积压
~0.3 s，队列中的旧帧永远无法通过 0.2 s 同步窗，造成连锁失败。现在
`capture_graspnet.Capture` 保留近 2 s 关节历史，取与图像时间戳**最接近**的
关节消息做同步判定——这才是"图像时刻关节状态"的正确语义，且对处理延迟
不敏感。重试顺序也改为最新帧优先。

## 8. 2026-11-08 执行层修订（重构轮）

审计驱动的执行层变更（详见 `docs/REFACTOR_REPORT_20261108.md`）：

- **候选复验**：所有候选（网络、旋转、几何）统一过 `grasp_candidates.
  revalidate_candidate`：局部宽度可行性（云+凸包取大者，防可见弧低估）、
  夹持体积包含、双侧接触支撑、预抓取腕部低于支撑面预检（上舀抓取的法兰
  在指尖后 142.5 mm，必然碰桌面，提前拒绝而不是烧掉 IK 尝试）。
- **渐进闭合**：固定 32 mm 闭合对 <32 mm 物体永远夹不到；固定过压又把
  小物体压到腕关节超差。现在从预测宽度+4 mm 起以 2 mm 步进合拢，双指
  接触即停，再锁紧一步（每边 1 mm）保证摩擦力高于滑移阈值。
- **附着提升**：提升规划前把目标以 `RobotState.attached_collision_objects`
  （本机 moveit_msgs 旧 schema，操作名 APPEND/REMOVE）附着到 gripper_link2，
  载荷参与碰撞预测；Gazebo 中仍由真实接触与摩擦决定成败。
- **障碍物**：观测中非目标簇输出为 AABB 障碍（遮挡物、第二物体），进
  MoveIt 规划场景；纯感知来源，不读场景真值。
- **分阶段计时与夹持状态分类**：报告含 `stage_timings_s` 与
  `grasp_state`（object_grasped / slipped / single_sided / empty_grip /
  blocked_unknown）。
- **端到端口径**：`execution_complete` = 物理抓取 + 保持稳定 + 末端反馈
  收敛，三者同时报告；批次 trial status 使用最严格的 execution_complete。
