# Piper / C1W 实机与 Gazebo 三维定位

更新：2026-10-18。本文覆盖颜色目标的相机坐标、基座坐标，以及已知 40 mm 直立红方块的中心位姿。颜色目标点是**可见表面的代表点**；方块位姿依赖已知尺寸和可见顶面，后续抓取仍需要 TCP、规划及执行反馈。

## 需要现场完成的事

1. 给 Piper 上电，接好 USB-CAN 与电脑；连接 OKULO-C1W 的电源和数据线。确认腕部支架、相机方向与现有 `link6 → camera_depth_frame` 标定时一致。
2. 准备一块哑光红色平面或红色小贴纸、硬板和标尺。首次测试保持机械臂固定，只移动红色目标。
3. 先验证相机坐标轴、深度变化和静止散差，再用独立测量的基座参考点验证手眼转换。完成这些记录之后再进入抓取测试。

设备未连接时，离线公式测试和 Gazebo 测试不能给出 C1W 实机的绝对定位精度。

## 已实现的数据链

```text
实机：Vidu RGB-D + 实时 CameraInfo/量程 ─┐
                                      ├→ camera_target → /piperx/target_camera
仿真：Gazebo RGB-D → camera_bridge ────┘                         │
                                               target_base ────┘
                                                     ↓
                                             /piperx/target_base
```

两个点话题都是 `geometry_msgs/PointStamped`，单位为米，并保留观测时间戳。

| 项目 | 实机 | Gazebo |
| --- | --- | --- |
| 彩色输入 | `/camera/color/aligned`，BGR `8UC3` | 同名同格式，由桥接节点转换 |
| 深度输入 | `/camera/depth`，`16UC1`；Z = raw × range / 65535 | 同名同格式，由 Gazebo `32FC1` 米深度转换 |
| 内参 | `/camera/info/depth`，设备实时 K/D | 同名，使用仿真相机自己的 K/D |
| 量程 | `/camera/depth_range`，设备元数据 | 同名，固定 3 m；无效及超范围深度置零 |
| 相机坐标系 | `camera_depth_frame`；X 右、Y 下、Z 前 | 同名光学坐标系 |
| 畸变 | C1W `kannala_brandt4`；也兼容 Brown-Conrady/plumb_bob | 理想针孔、零畸变 |
| 转到基座 | 实测手眼矩阵 × 观测时刻 `base_link → link6` TF | 观测时刻 `base_link → camera_depth_frame` TF |

Vidu 2.1 SDK 的 `example/deproject/main.cpp` 明确使用前四个 Kannala-Brandt 系数计算 theta 多项式；与 OpenCV fisheye 的反投影定义一致。本项目使用同一模型处理标定板角点与目标像素。`src/vidu_ros2_wrapper/src/vidu_ros2_wrapper.cpp` 从 SDK 点云的 Z 通道生成深度图，所以这里的深度是光轴 Z，不是到相机原点的欧氏距离，也不是直接以毫米计的 `16UC1`。

检测选取最大颜色连通区域，避开轮廓边界，在区域中选取实际有效像素，并从该像素附近的 5×5 区域取深度中位数。它能识别颜色区域，尚不区分同色物体类别，也不直接估计物体几何中心。

## 一条命令启动仿真

先按 README 获取上游依赖并编译，然后在仓库根目录运行：

```bash
bash scripts/start_gazebo_vision.sh
```

默认独立使用 `ROS_DOMAIN_ID=181`、`GAZEBO_MASTER_URI=http://127.0.0.1:11369`。脚本生成官方 Piper + 原厂夹爪模型，添加不可见的 link6 随动 RGB-D 传感器，启动只有地面和 **40 mm 红方块**的 `vision.world`，将仿真机械臂放到观察姿态，然后持续输出表面点、方块中心和顶面中心。Ctrl+C 结束并清理进程。启动前检测端口及 ROS 域冲突；需要多个实例时为每个实例指定不同的域与端口。

另一个终端打开三维窗口：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=181
export GAZEBO_MASTER_URI=http://127.0.0.1:11369
gzclient
```

相同环境下保存一帧或记录静止散差：

```bash
ros2 run piperx_lab camera_target --source sim --color red --once \
  --save-frame records/gazebo_vision/red_target.png --ros-args -p use_sim_time:=true
ros2 run piperx_lab target_check --frame base --source sim --samples 20 \
  --out records/gazebo_vision/repeatability.json --ros-args -p use_sim_time:=true
```

现有手眼彩色相机演示使用 `camera_bridge.py --color-only`；新 RGB-D 演示由启动脚本使用默认的 RGB-D 桥。相机中心静态展示仍使用独立的 `camera_center.world`，不提供视频流。

## 可复现的仿真几何验收

保持上述仿真运行，在同一 ROS 域执行：

```bash
python3 sim/gazebo/verify_vision.py
```

验证器依次改变红色方块位置、机械臂关节配置、第六轴角度及方块自身偏航角，读取实际发布的基座表面点与方块位姿，分别与 Gazebo 独立模型中的 40 mm 方块表面、中心及 90° 等价偏航角比较。每种情况默认各采集 5 点；表面距离和中心误差的仿真回归阈值均为 5 mm，偏航角阈值为 10°。原始点、仿真真值、误差和通过状态保存在 `records/gazebo_vision/verification.json`。验证器会移动仿真模型。Gazebo 的关节配置插件直接设置关节位置，这项测试验证不同静止姿态下的视觉坐标，不验证动力学轨迹控制或接触抓取。

该数值衡量点是否落在正确物体表面，不能当作物体中心误差或真实 C1W 精度。Gazebo 深度传感器不模拟 ToF 多径、反光、量程噪声或真实镜头畸变；鱼眼路径由 SDK 公式与离线已知点测试覆盖。

## 已知直立方块的视觉位姿

`cube_pose` 与 `camera_target` 共用对齐的 RGB-D 输入和红色区域分割。它将区域内有效像素按当前相机模型反投影成点云，并在观测时刻换到 `base_link`；只在顶面点足够、顶面尺寸与已知边长相符时，估计顶面中心和方块中心。模型假设方块已知边长、直立，且 `base_link` 的 +Z 与桌面法向一致。输出 `/piperx/cube_pose_base`（`PoseStamped`）和 `/piperx/cube_top_base`（`PointStamped`），单位为米。方块正方形顶面的偏航角只能确定到 **90° 等价类**；输出四元数是其中一个代表，不是有标记角点的完整六维物体朝向。该节点只读，不发布机械臂动作。

脚本已自动启动仿真模式；单独运行一次可用：

```bash
ros2 run piperx_lab cube_pose --source sim --size 0.04 --table-height 0 --once \
  --ros-args -p use_sim_time:=true
```

实机测试已知 40 mm 直立红方块时，在相机和机械臂反馈节点运行后执行：

```bash
ros2 run piperx_lab cube_pose --source real --size 0.04 --once \
  --save-frame records/c1w_acceptance/cube_pose_40mm.png
```

请另外量好方块实际边长和工作台的基座系高度；若提供 `--table-height`，节点会拒绝底部高度明显不符的结果。遮挡顶面、相机看不到足够顶面、同色多个物体或物体倾斜时，这个受限模型可能不给出位姿；不能将它当作通用六维姿态识别。当前实机位姿准确度仍待独立测量。

## 实机运行与验收

每个实机终端在仓库根目录先执行以下环境设置，使用默认实机域并关闭仿真时钟：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
```

终端 A 启动相机：

```bash
bash scripts/start_camera.sh
```

终端 B 持续识别红色目标并输出相机 XYZ：

```bash
ros2 run piperx_lab camera_target --source real --color red
```

先做三个只移动目标的检查：

1. 把红色平面放在清晰、有有效深度的位置，保持约 20 秒。检查 X 右移变大、Y 向画面下方移动变大、Z 远离相机变大。
2. 保持平面大致垂直光轴，在约 0.3 / 0.5 / 0.7 m 的位置分别观察；实际可用距离以设备有效深度为准。用标尺测量两次位置之间的位移，比较 Z 的变化量。镜头前表面与光学原点有偏移，从外壳起量只能做粗略绝对距离检查。
3. 在同一平面深度下，将目标放到画面中部、左侧、右侧，再分别记录。应检查边缘反投影是否异常；平面倾斜时 Z 本身会随位置变化，不能将其误判为畸变。

终端 C 为每个静止位置记录一次（文件名和 label 随位置更换）：

```bash
ros2 run piperx_lab target_check --frame camera --source real --samples 20 \
  --label center_static --out records/c1w_acceptance/center_static.json
ros2 run piperx_lab camera_target --source real --color red --once \
  --save-frame records/c1w_acceptance/center_static.png
```

记录工具按时间戳去重，拒绝错误坐标系、非有限数和超过 0.5 s 的时间偏差；记录所有采样点、均值、每轴标准差和最大静止散差。默认识别每秒输出一次，因此 20 点约需 20 秒；60 秒收不足会保存不完整结果并返回非零。

基座坐标还需要真实关节反馈与 TF。先确认 `can0` 已按当前机械臂配置正确连接，再在终端 D 启动驱动：

```bash
bash scripts/start_real.sh --can can0
```

终端 E 运行只读坐标转换：

```bash
ros2 run piperx_lab target_base --source real
```

`target_base` 默认加载 `calibration/handeye_result.json` 中通过质量检查的实机 Piper 外参，并读取 `base_link → link6`。在另一个终端记录：

```bash
ros2 run piperx_lab target_check --frame base --source real --samples 20 \
  --label base_static --out records/c1w_acceptance/base_static.json
```

如果已经通过夹具/独立测量获得**该可见表面代表点**在基座系中的 XYZ，可加 `--expected X Y Z`（米），输出参考点偏差和 RMSE；不要把物体中心或粗略尺量值当作高精度表面真值。

最后将一个小红色目标固定在基座附近的已知位置，现场操作者按已有流程改变腕部姿态，每次停稳后重新记录基座坐标。若检测对应的物理点保持一致，基座坐标也应保持一致。不同视角下代表点可能在物体表面移动，应先核对标注图，再区分视觉选点变化与手眼外参误差。现场需要更精确的同点验证时，应使用标定板角点或可重复定位夹具。

## 使用边界与故障定位

- `camera_target`、`target_base`、`target_check` 均只读；没有抓取或实机运动命令。仿真摆姿态脚本只面向 Gazebo 的轨迹话题。
- `target_base --source sim` 使用仿真 TF；`--source real` 使用实测外参。应分别位于不同 ROS 域，仿真节点需 `use_sim_time:=true`。
- 实机与仿真图像、深度及内参必须属于同一光学坐标系与同一观测时刻。当前两端桥接输出都保留匹配时间戳。
- 没有目标/有效深度时不发布点。TF 或目标过期会丢弃该帧；先排查相机/驱动频率、时钟与 TF，不能通过放宽时效门限掩盖误差。
- 深度/外参准确度的实测结果尚待设备连接后填写；当前实机手眼样本的内部一致性不能替代独立验收。
