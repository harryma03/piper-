#!/usr/bin/env bash
# Run the installed Vidu SDK ROS 2 wrapper for the connected Okulo camera.
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
exec ros2 run vidu_ros2_wrapper vidu_ros2_wrapper_node
