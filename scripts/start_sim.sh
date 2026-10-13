#!/usr/bin/env bash
# Piper MoveIt fake-hardware demo. No CAN driver is started.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EFFECTOR=agx_gripper
USE_RVIZ=true
NAMESPACE=
while (( $# )); do
  case "$1" in
    --effector)
      EFFECTOR="${2:?missing effector}"; shift 2 ;;
    --namespace)
      NAMESPACE="${2:?missing namespace}"; shift 2 ;;
    --no-rviz)
      USE_RVIZ=false; shift ;;
    -h|--help)
      echo 'Usage: start_sim.sh [--effector none|agx_gripper|revo2|revo2_pro|revo2_touch] [--namespace NAME] [--no-rviz]'
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$EFFECTOR" in
  none|agx_gripper|revo2|revo2_pro|revo2_touch) ;;
  *) echo "Unsupported effector: $EFFECTOR" >&2; exit 2 ;;
esac

set +u
source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export LC_NUMERIC=en_US.UTF-8
ARGS=(arm_type:=piper "effector_type:=$EFFECTOR" follow:=false "use_rviz:=$USE_RVIZ")
if [[ -n "$NAMESPACE" ]]; then
  ARGS+=("namespace:=$NAMESPACE")
fi
exec ros2 launch agx_arm_moveit demo.launch.py "${ARGS[@]}"
