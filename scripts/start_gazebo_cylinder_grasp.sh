#!/usr/bin/env bash
# Known-pose cylinder benchmark, isolated Gazebo domain. No real robot connection.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u
export ROS_DOMAIN_ID=183 GAZEBO_MASTER_URI=http://127.0.0.1:11371
export PYTHONPATH="$ROOT/src/piperx_lab:${PYTHONPATH:-}"
export PIPER_SIM_ARM_PID=1 PIPER_SIM_CONTROL_RATE=1000
export PIPER_SIM_MIN_CONTACT_DEPTH=0 PIPER_SIM_FINGER_KP=300 PIPER_SIM_FRICTION=2
export PIPER_SIM_OBJECT_SHAPE=cylinder PIPER_SIM_OBJECT_MASS=0.06
RUN_DIR="$ROOT/records/gazebo_grasp/cylinder_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
bash scripts/start_gazebo_grasp.sh > "$RUN_DIR/scene.log" 2>&1 &
scene_pid=$!
cleanup() {
  trap - EXIT INT TERM
  kill -TERM "$scene_pid" 2>/dev/null || true
  wait "$scene_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
for ((i=0; i<90; i++)); do
  if grep -q 'Isolated contact simulation ready' "$RUN_DIR/scene.log"; then break; fi
  if ! kill -0 "$scene_pid" 2>/dev/null; then cat "$RUN_DIR/scene.log"; exit 1; fi
  sleep 1
done
if ! grep -q 'Isolated contact simulation ready' "$RUN_DIR/scene.log"; then
  echo 'Gazebo startup timed out; see scene.log' >&2
  exit 1
fi
printf 'Cylinder grasp starting. Result: %s/result.json\n' "$RUN_DIR"
if python3 sim/gazebo/grasp/run_grasp.py --lift-scaling .02 --approach-scaling .03 \
    --grip-seconds 8 --hold-seconds 5 --reference-z -.11 --out "$RUN_DIR/result.json" \
    > "$RUN_DIR/action.log" 2>&1; then
  echo 'Cylinder lift completed; Gazebo remains open. Press Ctrl-C to close the simulation.'
else
  echo "Grasp did not pass; inspect $RUN_DIR/action.log. Gazebo remains open." >&2
fi
wait "$scene_pid"
