#!/usr/bin/env bash
# Dedicated domain and master: never starts a CAN driver.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u
export ROS_DOMAIN_ID=183
export GAZEBO_MASTER_URI=http://127.0.0.1:11371
export PYTHONPATH="$ROOT/src/piperx_lab:${PYTHONPATH:-}"
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models
python3 - <<'CHECK'
import socket
s=socket.socket();s.settimeout(.3)
occupied=s.connect_ex(('127.0.0.1',11371))==0
s.close()
if occupied:raise SystemExit('Grasp Gazebo master is occupied; stop the existing grasp simulation first')
import rclpy,time
from rclpy.node import Node
rclpy.init();node=Node('grasp_domain_guard')
end=time.monotonic()+1
while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
conflict=any(name in ('/joint_states','/feedback/joint_states','/control/joint_states') for name,_ in node.get_topic_names_and_types())
node.destroy_node();rclpy.shutdown()
if conflict:raise SystemExit('ROS domain 183 already contains a robot; stop it before starting this benchmark')
CHECK
OUT=/tmp/piper_contact_grasp
python3 sim/gazebo/grasp/make_scene.py "$OUT"
PIDS=()
cleanup(){ trap - EXIT INT TERM; for p in "${PIDS[@]}"; do kill -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done; sleep 1; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
setsid gzserver "$OUT/grasp.world" -s libgazebo_ros_init.so -s libgazebo_ros_factory.so > /tmp/piper-grasp-gazebo.log 2>&1 &
PIDS+=("$!")
setsid ros2 run robot_state_publisher robot_state_publisher "$OUT/robot.urdf" > /tmp/piper-grasp-rsp.log 2>&1 &
PIDS+=("$!")
ros2 run gazebo_ros spawn_entity.py -entity piper_contact -file "$OUT/robot.urdf"
for controller in joint_state_broadcaster arm_controller gripper_controller; do
 ros2 run controller_manager spawner "$controller" --controller-manager /controller_manager --controller-manager-timeout 45
done
setsid ros2 launch agx_arm_moveit move_group.launch.py arm_type:=piper effector_type:=agx_gripper follow:=false control_topic:=/joint_states > /tmp/piper-grasp-moveit.log 2>&1 &
PIDS+=("$!")
if [[ "${PIPER_SIM_HEADLESS:-0}" != 1 ]]; then
 setsid gzclient > /tmp/piper-grasp-gui.log 2>&1 &
 PIDS+=("$!")
fi
echo 'Isolated contact simulation ready on ROS domain 183'
wait "${PIDS[0]}"
