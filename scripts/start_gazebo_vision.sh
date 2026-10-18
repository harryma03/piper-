#!/usr/bin/env bash
# Isolated Gazebo RGB-D/3D-target demo. This script never starts the CAN driver.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"
set +u
source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export PYTHONPATH="$WORKSPACE/src/piperx_lab:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-181}"
export GAZEBO_MASTER_URI="${GAZEBO_MASTER_URI:-http://127.0.0.1:11369}"
export GAZEBO_MODEL_PATH="$WORKSPACE/sim/gazebo/models:/usr/share/gazebo-11/models"
export GAZEBO_RESOURCE_PATH="$WORKSPACE/sim/gazebo:/usr/share/gazebo-11"
export LC_NUMERIC=en_US.UTF-8

# Refuse an occupied master or ROS domain before publishing any sim command.
python3 - <<'PY'
import os, socket, time
from urllib.parse import urlparse
import rclpy
from rclpy.node import Node
uri = urlparse(os.environ['GAZEBO_MASTER_URI'])
if uri.scheme != 'http' or uri.hostname not in ('127.0.0.1', 'localhost') or not uri.port:
    raise SystemExit('Use a local http://127.0.0.1:PORT Gazebo master URI')
try:
    connection = socket.create_connection((uri.hostname, uri.port), timeout=0.3)
except OSError:
    pass
else:
    connection.close()
    raise SystemExit('Gazebo master port is occupied; choose another GAZEBO_MASTER_URI')
rclpy.init()
node = Node('piperx_vision_domain_check')
deadline = time.monotonic() + 1.0
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
conflict = any(name in ('/spawn_entity', '/feedback/joint_states', '/control/joint_states', '/tf')
               for name, _ in node.get_service_names_and_types() + node.get_topic_names_and_types())
node.destroy_node()
rclpy.shutdown()
if conflict:
    raise SystemExit('ROS domain already contains a robot or Gazebo; choose another ROS_DOMAIN_ID')
PY

MODEL=/tmp/piper_vision_${ROS_DOMAIN_ID}.urdf
python3 sim/gazebo/make_robot.py "$MODEL" --rgbd
RSP="$(ros2 pkg prefix robot_state_publisher)/lib/robot_state_publisher/robot_state_publisher"

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

gzserver sim/gazebo/worlds/vision.world -s libgazebo_ros_init.so -s libgazebo_ros_factory.so &
PIDS+=("$!")
"$RSP" "$MODEL" --ros-args -p use_sim_time:=true -r /joint_states:=/control/joint_states &
PIDS+=("$!")
python3 sim/gazebo/camera_bridge.py &
PIDS+=("$!")
ros2 run gazebo_ros spawn_entity.py -entity piper_vision -file "$MODEL"
kill -0 "${PIDS[0]}"
python3 sim/gazebo/set_vision_pose.py
python3 -m piperx_lab.target_base --source sim --ros-args -p use_sim_time:=true &
PIDS+=("$!")
python3 -m piperx_lab.camera_target --source sim --color red --ros-args -p use_sim_time:=true &
PIDS+=("$!")
python3 -m piperx_lab.cube_pose --source sim --size 0.04 --table-height 0 --ros-args -p use_sim_time:=true &
PIDS+=("$!")
echo "Gazebo 3D vision running: ROS_DOMAIN_ID=$ROS_DOMAIN_ID GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"
echo 'Press Ctrl+C to stop the demo.'
wait -n "${PIDS[@]}"
