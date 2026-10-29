# Piper 示例复现记录

更新：2026-10-28。目标是测试此前列出的 Piper 示例，重点验证仿真物理抓取。
所有执行均为离线、虚拟 CAN 或仿真；尚未执行实机抓取。

| 示例 | 当前证据 | 能证明的范围 |
|---|---|---|
| 原厂 pyAgxArm 夹爪接口 | 原厂虚拟 CAN 测试 2 项通过 | 开合命令、反馈、参数读写；没有接触物理 |
| 原厂 agx_arm_ros MoveIt 2 | 轨迹规划与执行通过；夹爪 40 mm / 0 mm 反馈匹配 | 模拟硬件的规划与控制；没有物体摩擦 |
| 原厂 agx_arm_sim MuJoCo 模型 | 原始场景与空载开合通过；另增加自由方块做 17 组夹持对照 | 有物体夹持证据；方块和控制轨迹是本地添加，不是上游现成完整抓取 demo |
| AgileX 教程收录的 GraspGen | 原工程配置构建失败，缺少 ROS 1 catkin；原始数值 IK 独立测试 6 项通过 | 局部运动学自洽；完整视觉抓取尚未运行 |
| AgileX 教程收录的 Isaac Lab 方块堆叠 | 原脚本 1/1 成功；诊断入口 6/6 成功，均通过独立末态位置检查 | 使用方块真值位置；不能推断视觉或实机成功率 |

Isaac Lab 数据采集工程和 GraspGen 均为 AgileX 教程收录的第三方工程，应与原厂 SDK 区分。

## 来源与版本

- [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)：`841a625f5f4920e776f20b934eb13048b747e6d0`。
- [agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros)：`e4ccec1999279d063c2962030b9437b78b0d0db6`。本地含项目补丁，见 `docs/UPSTREAM_PATCHES.md`；此次 fake-hardware 测试没有启用实机控制门。
- [agx_arm_sim](https://github.com/agilexrobotics/agx_arm_sim)：`f8cd8b147c75d59e14f90fb0646770eefa268ed0`。仅下载原始 Piper MuJoCo 模型与资产，每个文件均与 GitHub blob 哈希核对，记录在 `third_party/agx_arm_sim/SOURCE_MANIFEST.json`。
- [GraspGen](https://github.com/vanstrong12138/GraspGen)：`deaf1f5f5b5fcab32faf8742eb7ff2b641979f33`。下载运动学与视觉模块源码子集，哈希清单在 `third_party/GraspGen/SOURCE_MANIFEST.json`，未下载推理权重与设备驱动。
- [IsaacLab_Data_Collection](https://github.com/szyzp/IsaacLab_Data_Collection)：`574b48562c2dffe301796a2ba39d2ae22a9c8b4c`。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2)：`37ddf626871758333d6ed89cf64ad702aef127d0`，配套 Isaac Sim 5.1.0、Python 3.11、Torch 2.7.0+cu128。

## 已执行检查

### SDK 虚拟 CAN

直接运行 SDK 原有的 `tests/test_agx_gripper_virtual_can.py`，使用 python-can 的 `virtual` 接口，不是 `can0`。

结果：2 passed。原始 JUnit 报告：
`records/official_demo_tests/sdk_gripper_virtual_can.xml`。

### MoveIt 2

在独立 `ROS_DOMAIN_ID=185`、`/official_demo` 命名空间启动 `scripts/start_sim.sh --no-rviz --namespace official_demo`。
检查器先通过控制器管理服务确认硬件插件为 `mock_components/GenericSystem`，再请求真实 MoveIt 规划服务与执行 action。

- 手臂第一关节目标 0.1 rad，生成 10 个轨迹点，执行结果码 1。
- 各关节反馈与目标误差小于 0.002 rad。
- 夹爪打开至 0.040 m、闭合至 0.000 m，两次 action 成功，新的关节反馈匹配。
- 测试后已停止演示进程。

检查器：`sim/official_demos/check_moveit_demo.py`。
结果：`records/official_demo_tests/moveit_fake_result.json`。

### MuJoCo

使用 MuJoCo 3.3.7，原样加载上游 `scene.xml` 与 `piper.xml`，从 `home` 状态启动。
每段运行 3 s，共 4500 步；单指目标为 0 → 0.03 → 0 m，双指位置分别跟踪正负目标，状态保持有限值。

读取到的夹指碰撞面摩擦参数为 `[1.0, 0.005, 0.0001]`，依次对应滑动、扭转和滚动摩擦；这是该 MuJoCo 模型实际使用的默认参数，并非厂商实测接触材料参数。
夹爪执行器限制为 `[-10, 10]`，位置增益 40，夹指采用盒状碰撞面。

检查器：`sim/official_demos/check_mujoco_model.py`。
结果与截图：`records/official_demo_tests/mujoco_piper_result.json`、`mujoco_piper.png`。

在不改原模型的前提下，本地诊断又添加了一个自由刚体方块（边长 40 mm、质量 80 g）和桌面。每次试验让已张开的夹指位于方块两侧，闭合 2 s，再以同一条关节轨迹抬升 3 s，最后保持 1 s。方块没有通过约束与夹爪绑定。下表中摩擦值同时设置于方块与夹指，单次保持成功要求最后 0.5 s 内方块高于初始位置 50 mm 且两指均有非零法向接触力。

| 位置驱动 Kp | 滑动摩擦系数 | 最大抬升 | 最后 0.5 s 持握 | 闭合结束时两指法向力之和 |
|---:|---:|---:|---|---:|
| 40（模型默认） | 1.0（模型默认） | 8.7 mm | 失败，方块回落 | 约 0.78 N |
| 80 | 1.0 | 55.5 mm | 通过，51/51 采样点 | 约 1.55 N |
| 160 | 1.0 | 55.5 mm | 通过，51/51 采样点 | 约 3.00 N |
| 40 | 2.0 | 55.5 mm | 通过，51/51 采样点 | 约 0.78 N |
| 160 | 0.5 | 55.5 mm | 通过，51/51 采样点 | 约 3.00 N |

这组对照支持“该场景中有效夹持力与摩擦余量不足可能导致滑落”，但不能单独证明当前 Gazebo 圆柱滑落的原因，也不代表这些参数适合真实夹爪。方块一开始已经处于夹指之间；没有测试视觉、接近轨迹或误差。每种配置仅进行一次确定性试验。

诊断使用 `mujoco.mj_contactForce` 分别读取每个接触点的法向力和切向力；记录的是同一夹指各接触点法向力的标量和，以及各点切向力大小的标量和，并非切向力的矢量合力。没有可靠接触面积，不能由这些数值推算压力。Kp 是位置控制增益，既不是夹持力的直接设定值，也不是厂商实测值。

额外改变方块质量做了四次对照：默认 Kp 40、摩擦 1.0 时，40 g 方块保持成功；Kp 80、摩擦 1.0 时，120 g 方块保持成功；Kp 40、摩擦 2.0 时，120 g 方块保持成功；Kp 160、摩擦 0.5 时，160 g 方块滑落。这些仍是单次确定性试验。

对 80 g 方块、Kp 80、摩擦 1.0，另外测试了 8 个确定性初始偏差：世界坐标 X 或 Y 单轴各 ±5 mm、两个同号对角偏移各 ±5 mm，以及方块绕世界 Z 轴转动 ±15°。8 次均满足最后 0.5 s 的双指接触与抬升条件，最终抬升 54.0–55.4 mm。此处方块仍预先放在张开的夹指之间，不代表完整视觉抓取对识别偏差的成功率。

对基准、Kp 80 和摩擦 2.0 三种条件又各复测一次，并增加方块相对夹爪的位置与方块姿态记录。基准条件下，方块从闭合结束到末态相对夹爪移动约 63.35 mm；Kp 80 与摩擦 2.0 两种条件分别约移动 0.56 mm 和 0.57 mm，末尾 0.5 s 的最大相对位移约 0.07 mm。新增记录的成功条件还要求末尾 0.5 s 的相对位移小于 5 mm。这些数字只说明该模拟方块与轨迹下的表现。

对 80 g 的默认实验，方块重力约为 0.785 N，闭合末两指法向力之和约 0.784 N。理想静态条件下，摩擦系数为 1 时可提供的摩擦支撑力已接近重力，抬升加速度、姿态和接触变化都可能使余量不足；这个估算用于解释对照趋势，不是对动态滑落原因的完整证明。

检查器：`sim/official_demos/mujoco_cube_diagnostic.py`。每组的 `result.json`、`trajectory.csv` 和末态截图分别保存在 `records/official_demo_tests/mujoco_cube_*/`。

### GraspGen

配置原始 `tensorrtx_yolov8_ros/CMakeLists.txt` 时，编译器检查通过，随后因缺少 `catkinConfig.cmake` 失败。
原工程依赖 ROS 1 Noetic、RealSense、CUDA/TensorRT 等环境；当前工作区为 ROS 2 Humble / Okulo C1W。
没有将该构建失败解释为算法失败，也没有运行相机或实机驱动。

源码的视觉执行入口是 `yolov8_seg.cpp`，结合分割结果与深度生成物体点云；抓取姿态相关代码在 `AABBGraspPlanner.hpp`。

独立编译并调用默认 ROS 节点使用的原始 `PiperJacobianIK` 与 `PiperForwardKinematics` 头文件，使用节点相同的 modified DH、关节限制、迭代次数和容差。
对 6 组可达关节姿态执行 FK → 扰动初值 → IK → FK 检查，6 项通过，位置误差为 0.330–0.914 mm。
这仅验证这套 FK/IK 的局部一致性，没有验证与实物尺寸、相机标定或识别结果的一致性。

构建记录：`records/official_demo_tests/graspgen_configure.log`。
运动学结果：`records/official_demo_tests/graspgen_ik_result.json`。

## Isaac Lab 当前状态与检查方法

运行环境独立保存在 `.isaac-venv`，启动时清除继承的 ROS Python/动态库搜索路径，不修改 ROS 工作区的 Python 环境。
Isaac Lab 与 Piper 任务包已成功安装；原始 `flatdict==4.0.1` 在新版构建工具中缺少 `pkg_resources`，已通过 `sim/isaac/build_constraints.txt` 将构建工具限定为 setuptools 80.9.0，保持上游依赖版本不变。
Isaac Sim 5.1.0 的组件也已安装，但包元数据检查仍报告两处版本冲突：Isaac Lab 声明 starlette 0.49.1，而 Isaac Sim 安装了 0.45.3；IPython 9.17.1 要求 psutil 7，而 Isaac Sim 安装了 5.9.8。具体影响尚需运行验证，不能把环境描述为依赖检查全部通过。记录见 `records/isaac_stack/pip_check.log`。
首次空场景启动在下载扩展时达到 600 s 时限；复用缓存的第二次启动通过官方 `create_empty.py` 的 20 个物理步，结果为 `records/isaac_stack/smoke_result.json`。随后开始 Piper 方块任务，数据记录在 `records/isaac_stack/piper_trial_20261028/`。
三个 PyTorch wheel 的 SHA256 均与官方索引一致，核对记录在 `records/isaac_stack/wheel_checksum.log`。
`torch_gpu_check.json` 记录 RTX 5060、CUDA 12.8、计算能力 12.0 与成功的 GPU 矩阵运算。

USD 检查包含实例内部几何，共遍历 761 个 prim，没有找到显式接触摩擦属性。
两指原始线性驱动上限为 10、刚度 100、阻尼 1；任务 Python 配置会覆盖为上限 22、刚度 800、阻尼 20。
高 PD 配置同时关闭机械臂重力，不能直接把它的表现等同于当前 Gazebo 设置或真实夹持力。
USD 存在一条 `/visuals/world` 引用解析警告，需在实际启动后继续核对。
完整审计：`records/isaac_stack/piper_usd_audit.json` 与同目录 `.log`。

待运行的原始任务：`Isaac-Stack-Cube-Piper-IK-Rel-v0`。
它使用仿真方块真值位置、相对位姿 IK 和二值夹爪位置命令，不包含视觉识别。
`sim/isaac/run_cube_grasp.py` 复用原始控制器和成功条件，限定轮次、记录方块高度、末端位姿、夹指位置、执行器作用力估算及实际材质参数。
该模型使用隐式执行器；Isaac Lab 的 `applied_torque` 是经过截断的 PD 估算，不是实测关节作用力或指尖接触法向力。高度与闭爪命令同时出现也不能单独证明稳定夹持。

初始随机种子 42 的一轮试验，两个方块分别最高上升 251.5 mm 和 274.0 mm，最终高度差为 46.81 mm、46.80 mm，水平偏差为 3.56 mm、0.66 mm。末尾连续 10 个采样点夹爪均已张开，三个方块的最大位置变化分别约 0.024、0.062、0.103 mm。两个被抓方块在闭爪且高出初始位置 50 mm 的最长连续区间内，相对末端的最大位置变化约 0.031 mm 和 0.629 mm。结果由 `sim/isaac/analyze_cube_run.py` 独立计算；仍没有直接测量两指法向接触力。

同一诊断入口以随机种子 123 追加 5 个不同初始位置：原始成功条件和独立的末态检查均为 5/5。独立检查要求相邻方块的水平偏差小于 10 mm、高度差与 46.8 mm 的偏差小于 5 mm，且末尾 10 个采样点夹爪张开、每个方块位移小于 3 mm。实际最终水平偏差为 1.74–8.10 mm；各次闭爪抬升区间内，被抓方块相对末端的最大位移为 0.019–1.737 mm。上述 6 轮数据集分别有 1 条和 5 条演示，保存在各自的 `upstream_dataset.hdf5` 中。

此外直接运行未修改的 `scripts/tools/record_ik_stack.py` 一次，原程序报告 1 条成功演示。独立读取 `records/isaac_stack/upstream_unmodified/dataset.hdf5`，两处最终高度差分别为 46.82 mm、46.80 mm，水平偏差分别为 4.46 mm、1.44 mm；末尾 10 个采样点夹爪张开且方块位置稳定。独立分析为 `independent_analysis.json`。原始记录器在每步**之前**保存观测、在每步**之后**保存刚体状态，计算方块相对末端的位移时已将两者对齐；闭爪抬升区间最大相对位移约 0.032 mm、1.098 mm。这次原脚本未指定随机种子。所有结果仍没有直接测量 Isaac 中两指的法向接触力。

从 PhysX 视图读取的材料参数：机器人碰撞形状均为静摩擦 0.5、动摩擦 0.5；三个方块均为静摩擦 0.8、动摩擦 1.0、质量 20 g。这些是仿真中的单个形状参数，不等同于两种材料最终组合后的接触摩擦系数或实物测量值。夹指的隐式执行器设置为刚度 800、阻尼 20、`effort_limit_sim=22`，机械臂的高 PD 配置关闭了机械臂重力，方块仍受重力；这些条件与前面的 MuJoCo 80 g 方块实验不同。

启动入口：

```bash
bash scripts/start_isaac_cube_grasp.sh
```

默认单环境、无界面、1 轮、最多 700 个控制步、600 s 外部超时；系统支持时对子进程设置 8 GiB 内存上限。
输出到 `records/isaac_stack/<时间>/`，包括 `run.log`、`trajectory.csv`、`result.json`，若原始成功条件满足则导出数据集。
退出码 0 仅表示原始成功条件触发，仍需结合轨迹审核实际抓取过程。

源码中发现的待验证事项：轨迹生成忽略末尾一段时长，并从第一个预抓取点开始插值；第一次闭爪命令与上抬段同时开始。
已在不启动物理仿真的条件下执行原始控制器的轨迹生成：共生成 460 个控制步，第 60 步第一次闭爪，下一步目标高度已增加约 4.76 mm。记录在 `records/isaac_stack/upstream_trajectory_audit.json`；这仍不能代替物理抓取试验。
原始堆叠成功条件对高度差只做单侧比较，因此后续必须结合实际高度和持握过程判断，不能只看 success 日志。
当前没有修改上游控制器；这 7 次成功不代表对所有初始位置、真实物体或视觉识别都成功。

## 在新工作区恢复上游源码

以下目录和本机仿真环境、原始运行记录均按 `.gitignore` 保留在本地。本文所列 `records/` 路径是运行这些实验的机器上的记录，仓库不包含它们。仿真结果摘要见上文。

进入仓库根目录后，可通过已提交的清单恢复 MuJoCo 与 GraspGen 示例所需的上游源码子集。脚本逐文件核验 Git blob 哈希：

```bash
python3 - <<'PYCODE'
import json, subprocess, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix="piper-source-tree-") as directory:
    for name in ("agx_arm_sim", "GraspGen"):
        manifest = json.loads((Path("third_party") / name / "SOURCE_MANIFEST.json").read_text())
        tree = {"sha": manifest["commit"], "tree": manifest["files"], "truncated": False}
        path = Path(directory) / (name + ".json")
        path.write_text(json.dumps(tree))
        subprocess.run(["python3", "sim/official_demos/fetch_source_subset.py", name, "--tree", str(path)], check=True)
PYCODE
```

Isaac Lab 和任务代码是独立的 Git 仓库，首次安装时可在空目录中获取文首列出的固定版本：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
git -C third_party/IsaacLab checkout 37ddf626871758333d6ed89cf64ad702aef127d0
git clone https://github.com/szyzp/IsaacLab_Data_Collection.git third_party/IsaacLab_Data_Collection
git -C third_party/IsaacLab_Data_Collection checkout 574b48562c2dffe301796a2ba39d2ae22a9c8b4c
```

这只恢复源码；还需按对应上游说明安装 Isaac Sim、Isaac Lab、Python 环境、GPU 驱动和任务包。不同机器的版本与硬件兼容性需要各自验证。
