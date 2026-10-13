# 本地上游改动

本工作区通过 `upstream.repos` 固定 AgileX `agx_arm_ros`（`e4ccec1`）和 `pyAgxArm`，并保留 PointSpread Vidu 2.1 ROS 2 wrapper 的本地源码副本。AgileX 改动另保存为 `patches/agx_arm_ros_local.patch`，供新 checkout 应用。更新上游源码时需重新核对以下改动：

1. `src/agx_arm_ros/src/agx_arm_moveit/launch/demo.launch.py`：自动控制门的 `status_topics` 增加 `gripper_controller/follow_joint_trajectory/_action/status`。原版只看手臂 action，夹爪单独执行时实机控制门不会打开。这个变更已在本地代码中完成，但尚未经过实机验证。
2. `src/vidu_ros2_wrapper/src/vidu_ros2_wrapper.cpp`：增加 `std_msgs/Float32` 话题 `/camera/depth_range`，发布 SDK 帧元数据 `Range`；缺少有效量程时停止深度发布。原版将深度换算为 `16UC1`，却没有发布换算所需的量程。先前测试的 Okulo C1N 量程约 2.4983 m；当前 C1W 必须使用实时发布的量程。另移除无订阅／服务节点不需要的每帧 `rclcpp::spin_some(node)`，避免 Ctrl+C 后在已关闭的 ROS 上下文里创建 executor guard condition。

`src/piperx_lab` 是本项目自有代码，不属于上游补丁。
