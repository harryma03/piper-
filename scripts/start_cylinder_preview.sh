#!/usr/bin/env bash
# Uses existing real camera/driver/MoveIt. Only planning services are called.
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"
set +u
source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export PYTHONPATH="$WORKSPACE/src/piperx_lab:${PYTHONPATH:-}"
PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
python3 -m piperx_lab.cylinder_target --base --samples 300 --timeout 600 \
  --out records/cylinder_real/preview_observations.json > /tmp/piper-cylinder-observer.log 2>&1 &
PIDS+=("$!")
ros2 launch agx_arm_moveit moveit_rviz.launch.py arm_type:=piper \
  effector_type:=agx_gripper follow:=true \
  "rviz_config:=$WORKSPACE/config/cylinder_preview.rviz" > /tmp/piper-cylinder-preview-rviz.log 2>&1 &
PIDS+=("$!")
python3 -m piperx_lab.cylinder_preview --plan --keep-alive --object-spec "$WORKSPACE/config/cylinder_object.json" "$@"
