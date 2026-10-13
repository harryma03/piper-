# 提交到 `harryma03/piper-` 的目录与路径约定

2026-10-13 查看 [目标仓库](https://github.com/harryma03/piper-) 时仓库为空，尚无默认分支或现有目录需要兼容。以下布局以仓库根目录作为 ROS 2 工作区根目录；克隆到任意用户名、任意目录名均可。

提交对应的实机为 **AgileX Piper 六轴机械臂 + 原厂 `agx_gripper` 夹爪**，腕部相机为 **PointSpread OKULO-C1W-H106/940**。`ROBOT_PIPERX` 只是旧工作区名；驱动按 `arm_type:=piper` 配置，实机手眼外参对应 `link6 → camera_depth_frame`。

```text
piper-/
├── .gitattributes
├── .gitignore
├── README.md
├── upstream.repos                 # 固定 AgileX 驱动和 pyAgxArm 版本
├── patches/agx_arm_ros_local.patch
├── calibration/                  # 200 mm 板、12 组实机样本和实机外参
├── docs/
├── output/pdf/                   # 可原尺寸打印的标定板
├── records/gazebo_piper_handeye/ # Piper 仿真验证资料
├── scripts/
├── sim/
└── src/
    ├── piperx_lab/               # 自有 ROS 2 包，包含板资源副本
    ├── vidu_ros2_interfaces/     # PointSpread wrapper 接口源码
    └── vidu_ros2_wrapper/        # PointSpread wrapper 源码及本地补丁
```

`src/agx_arm_ros/` 和 `src/pyAgxArm/` 是外部 checkout，不作为普通文件夹提交到师兄仓库。本机 `src/agx_arm_ros/` 本身有 `.git`，直接 `git add` 会形成嵌套仓库引用，师兄克隆时不会自动得到可用源码。`sdk/`、`build/`、`install/`、`log/`、旧 `samples_old/` 和打包文件也不提交；根目录 `.gitignore` 已排除它们。AgileX 的本地代码改动在 `patches/agx_arm_ros_local.patch`，其中包含控制门话题调整和脚本执行权限。

## 新机器克隆后的步骤

```bash
git clone https://github.com/harryma03/piper-.git
cd piper-
vcs import src < upstream.repos
git -C src/agx_arm_ros submodule update --init --recursive
git -C src/agx_arm_ros apply --check ../../patches/agx_arm_ros_local.patch
git -C src/agx_arm_ros apply ../../patches/agx_arm_ros_local.patch
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
python3 -m pytest src/piperx_lab/test -q
ros2 run piperx_lab handeye_solve --samples calibration/samples --out /tmp/piper_handeye_review.json
```

`vcs` 来自 `python3-vcstool`。相机 wrapper 还要求目标机器单独安装 PointSpread Vidu 2.1 SDK，让 CMake 找到 `Vidu_SDK_Api`；它不是这个 Git 仓库的一部分。`pyAgxArm` 则在 `upstream.repos` 中固定到 `841a625f`，由 `colcon` 从 `src/pyAgxArm` 构建。

运行脚本从自身位置找到工作区，ROS 包从安装索引找到官方 URDF。`handeye_capture`、`handeye_solve` 和 `target_base` 的默认实机数据路径由仓库根目录定位，不再取决于终端当前目录。若使用非 symlink 安装并在仓库外运行，可设 `PIPER_LAB_WORKSPACE=/实际/克隆/目录`。`calibration/samples/` 与 Piper 仿真样本中的 `board_json`、`color_image`、`detected_image` 均相对各自 JSON 文件；搬迁后仍可打开。`calibration/handeye_result.json` 的 `source_samples` 是相对仓库根目录的记录，仅作溯源，求解和坐标转换不依赖这些路径。

当前 GitHub 仓库没有分支，因此尚不能检查仓库分支保护或与师兄已有代码的冲突。本文件只准备本地提交结构；尚未向远端推送。

目标仓库的页面目前无需登录即可访问。上述提交布局包含 12 组实机彩图和标注图，以及相机序列号；正式推送前应由项目成员确认这些资料适合公开。
