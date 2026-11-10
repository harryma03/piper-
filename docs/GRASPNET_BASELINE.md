# GraspNet Baseline 离线试验

这是一项抓取候选可视化试验，不发送 ROS 控制命令。候选位于相机坐标系，采用 GraspNet 自己的夹爪坐标约定；尚未转换成 Piper TCP，也未做夹爪尺寸、碰撞、逆解和完整轨迹验证。

## 来源和环境

- 官方源码：<https://github.com/graspnet/graspnet-baseline>，本机版本 `280c215129f759ed8649cb4e89fc5dfee55f4f80`，本机目录 `third_party/graspnet-baseline/`。
- 权重：官方 Google Drive 下载受配额限制，本次从 <https://huggingface.co/dgrachev/a2_pretrained> 获取第三方发布的 `checkpoint-rs.tar`。SHA-256 为 `60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868`，与该发布页一致；官方没有公布用于对照的校验值，因此不能确认和官方文件完全一致。模型以 PyTorch 的 `weights_only=True` 加载，并严格校验网络参数。
- 本次环境：RTX 5060 Laptop；`.graspnet-venv` 通过 `.pth` 引用本机 `.isaac-venv` 中的 PyTorch 2.7.0 + CUDA 12.8 等包；独立的 CUDA 12.8 编译组件位于 `third_party/graspnet-cuda-12.8/`。这两个目录和编译产物均不提交，不应假定其他电脑已具备同一环境。
- 官方项目说明其数据、代码和模型供免费非商业使用；其他用途须遵守原项目许可。
- 上游的 `torch.cross` 未指定向量维度，候选数恰好为 3 时可能得到错误的旋转矩阵。本次已在本地源码应用 `patches/graspnet_baseline_cross.patch`；重新检出源码后需要在源码目录执行 `git apply ../../patches/graspnet_baseline_cross.patch`。

## 运行

在项目根目录：

```bash
.graspnet-venv/bin/python scripts/graspnet_offline_demo.py \
  --example --checkpoint third_party/graspnet-baseline/checkpoints/checkpoint-rs.tar \
  --output records/graspnet_baseline/official_example

.graspnet-venv/bin/python scripts/graspnet_offline_demo.py \
  --input records/cylinder_real/geometry_diagnostic/frame.npz \
  --roi-from-red-mask 40 --roi-depth-window 0.08 \
  --checkpoint third_party/graspnet-baseline/checkpoints/checkpoint-rs.tar \
  --output records/graspnet_baseline/c1w_roi40_depth08
```

实机录制帧直接使用 C1W SDK 导出的 XYZ 点云，不把带鱼眼畸变的深度图当作针孔图重新反投影。红色掩码用于给推理输入限定图像区域，保留区域上下文；并非物体的完整分割。候选仅经过数值有效性和显示去重检查，模型分数不是成功概率。结果包括 `result.json`、全部候选数组和可视化图片，存放在被忽略的 `records/graspnet_baseline/`。

2026-11-04 的离线试验中，官方样例生成 342 个原始候选；C1W 录制帧使用红色区域外扩 40 像素和可见红色点深度中位数前后 8 cm 的输入窗口时，生成 23 个原始候选，数值检查后保留 23 个，显示去重后 7 个。窗口外扩 80 像素、相同深度范围时，网络未产生候选。两者都没有完成碰撞或可达性检查，候选数量不能用于估算抓取成功率；其对区域范围的敏感性仍需调查。

## Gazebo 视觉抓取试验

2026-11-06 在 ROS 域 183、Gazebo 私有 master 的单次仿真中，模拟 RGB-D 图像、关节状态和图像时刻 TF 来自同一帧；TF 时间差为零。红色圆柱拟合使用直径 46 mm、高 80 mm、竖放和桌面高度先验，模型推理不读取场景生成器的坐标。网络 230 个候选经开口和红色观测点筛选后，保留一个模型候选的三个预抓取距离。该候选直接使用时，6 cm 预抓取点未找到逆解。

已知该圆柱绕竖直轴对称，于是以观测到的圆柱中心为轴，将最高分候选绕轴采样 16 个角度；这是假定圆柱完整且对称的几何扩展，并非网络重新预测，也未单独验证旋转后的红色表面接触。67.5° 候选经预抓取、接近及不含附着物体的手臂抬升预规划后在物理仿真执行，双指在闭合和抬升期间均有接触；物体抬高 95.9 mm，最后 10 个采样的最大倾斜约 0.94°、最大位移约 0.18 mm，通过本项目现有成功判据。观察位置与仿真评估真值偏差约 1.15 mm，其中高度使用任务先验。完整结果位于本机 `records/graspnet_gazebo/symmetry_try2/grasp_result.json`，记录目录不随仓库提交。

这只是一处位置的一次成功，不能估算总体成功率或实机效果。预规划抬升没有把物体作为附着碰撞体；接近阶段最后最多 10 mm 允许目标与双指碰撞。GraspNet 夹爪和 Piper 夹爪几何不同，转换以夹指端部对齐作为近似。相机、机器人动力学与夹持参数都是理想化仿真，权重来源限制见上文。

在已启动的独立 Gazebo 场景中，可按顺序采集、暂停、推理、构造候选、恢复并执行（场景启动参数见 `docs/GAZEBO_VISION_GRASP.md`）。采集后应立即暂停，使模型推理期间仿真时间不流逝：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=183 ROS_LOCALHOST_ONLY=1 GAZEBO_MASTER_URI=http://127.0.0.1:11371
export PYTHONPATH="$PWD/src/piperx_lab:$PYTHONPATH"
OUT=records/graspnet_gazebo/new_trial
python3 sim/gazebo/grasp/capture_graspnet.py --out "$OUT/frame.npz"
ros2 service call /pause_physics std_srvs/srv/Empty '{}'
.graspnet-venv/bin/python scripts/graspnet_offline_demo.py --input "$OUT/frame.npz" --input-kind gazebo --roi-from-red-mask 30 --roi-depth-window 0.08 --checkpoint third_party/graspnet-baseline/checkpoints/checkpoint-rs.tar --output "$OUT/inference"
.graspnet-venv/bin/python scripts/prepare_graspnet_sim_candidates.py --frame "$OUT/frame.npz" --metadata "$OUT/frame.json" --grasps "$OUT/inference/grasps_all.npy" --out "$OUT/candidates.json"
python3 scripts/augment_cylinder_grasp_symmetry.py --candidates "$OUT/candidates.json" --observation "$OUT/frame_observation.json" --out "$OUT/symmetry.json"
ros2 service call /unpause_physics std_srvs/srv/Empty '{}'
python3 sim/gazebo/grasp/run_grasp.py --vision-observation "$OUT/frame_observation.json" --graspnet-candidates "$OUT/symmetry.json" --grip-seconds 8 --hold-seconds 5 --approach-scaling .03 --lift-scaling .02 --out "$OUT/grasp_result.json"
```

如果候选为空、观测时间过期或 TF 与图像时间不符，流程应停止；不要用仿真真值回填。输出的 GraspNet 候选仅供上述隔离仿真流程测试，不可直接发往实机。

## 随机位置与方块试点（2026-11-06）

新增 `scripts/run_gazebo_graspnet_batch.py`，每个位置独立重启 Gazebo，依次执行采集、暂停、多个 ROI 推理、目标开口/点云筛选、姿态扩展、MoveIt 预规划和物理接触测试。生成位置只用于场景设置和最终评估，识别与规划输入仍是 RGB-D 图像、同一时刻 TF 和关节状态。

固定 46 mm 直径、80 mm 高圆柱在 `x=0.40–0.50 m, y=-0.05–0.05 m` 内随机放置 10 个位置，`seed=20261106`。按物理抓取判据，8/10 完成双指接触并把物体稳定抬起至少 70 mm；其中 5/10 也完成了严格的末端轨迹反馈容差。2/10 没有任何候选通过预抓取、接近和不含附着物体的抬升检查。10 次视觉中心误差为 1.079–1.300 mm。原始批次曾把“轨迹反馈未完全收敛”和“未抓起”合并为失败，后续已拆开。结果为 `records/graspnet_gazebo/random10_seed20261106/summary.json`。

随后增加红色上方方块观测分支：使用红色点云估计水平范围并保持桌面、边长和直立先验，相机真值仍未进入规划。36–144 个候选通过开口和目标关联，但两个种子下的 6 次纯 GraspNet 方块试点没有任何候选通过预抓取逆解。为此增加显式 `--geometry-fallback`：只在全部模型候选失败后，使用观测中心和方块任务高度生成几何上抓候选，并保留 `kind=geometric_symmetry_fallback`、`fallback=true` 标记。

使用几何回退的 3 个随机方块位置按物理抓取均成功，物体分别抬升 97.42、96.68 和 93.43 mm，双指在闭合与抬升阶段均有接触；视觉中心误差为 0.929–1.916 mm。第三个位置仅末端关节反馈未完全收敛，物理上已经抓起。结果在 `records/graspnet_gazebo/cube_pilot_seed403/summary.json`。因此，当前方块成功来自相机定位加几何回退，不能表述为 GraspNet 直接生成了可用姿态。

这些结果仍受固定上方视角、圆柱/方块先验、理想模拟相机、物体重力与夹持参数未做实机标定、以及抬升预规划不含附着碰撞体等限制。尤其不能把 8/10 或 3/3 当作实机成功率。代码已覆盖随机位置和两种基本形状，但尚不能称为任意随机物体抓取。

2026-11-06 又用 `--xy` 固定重跑了 7 个历史失败位置：早先随机批次第 4、10 例，以及 GraspNet 随机批次第 1、5、6、7、9 例。按物理抓取判据，6/7 已经抓起并完成 70 mm 以上抬升；只有 1 例仍有 32 个候选全部未通过预抓取/Cartesian 检查。物理成功的六例中，两例末端反馈完全收敛，另外四例 joint4/joint5 的末态误差约 0.020–0.033 rad，未进入 0.015 的严格容差。该结果说明此前把反馈超时误记为未抓起；不能据此说末端控制已经完全达到目标。记录在 `records/graspnet_gazebo/retest_prior_failures_valid/summary.json`。

随后把最早规则几何 `random_seed42` 使用的 10 个位置全部用当前 GraspNet 流程固定重跑。按物理抓取判据，规则几何为 10/10，GraspNet 为 9/10；若只统计关节反馈完全收敛，则规则几何为 8/10，GraspNet 为 6/10。逐例对照：

| 位置 | 规则几何（物理） | GraspNet（物理） | GraspNet 反馈/失败说明 |
| ---: | :---: | :---: | --- |
| 1 | 成功 | 成功 | 已抬起；joint4/joint5 反馈未完全收敛 |
| 2 | 成功 | 成功 | — |
| 3 | 成功 | 成功 | 已抬起；joint4/joint5 反馈未完全收敛 |
| 4 | 成功 | 成功 | 已抬起；joint4/joint5 反馈未完全收敛 |
| 5 | 成功 | 失败 | 32 个候选均未通过预抓取/Cartesian 检查 |
| 6 | 成功 | 成功 | — |
| 7 | 成功 | 成功 | — |
| 8 | 成功 | 成功 | — |
| 9 | 成功 | 成功 | — |
| 10 | 失败 | 成功 | — |

九个 GraspNet 物理成功位置的最终抬升为 87.52–96.90 mm，视觉中心误差为 1.074–1.279 mm；唯一未抓起的是第 5 例，原因是候选姿态未通过完整路径检查。规则几何和 GraspNet 在这十个位置上物理成功率分别为 10/10 和 9/10；末端反馈完全收敛率则为 8/10 和 6/10。记录在 `records/graspnet_gazebo/geometry_seed42_retested_with_graspnet/summary.json`。

复现随机位置批次：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --trials 10 --seed 20261106 \
  --out records/graspnet_gazebo/random10_seed20261106

python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape cube --trials 3 --seed 403 \
  --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/cube_pilot_seed403
```
