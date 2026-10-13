#!/usr/bin/env bash
# Piper real-arm driver + MoveIt. The arm is not auto-enabled by default.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAN_PORT=
EFFECTOR=agx_gripper
AUTO_ENABLE=false
NAMESPACE=
while (( $# )); do
  case "$1" in
    --can)
      CAN_PORT="${2:?missing CAN interface}"; shift 2 ;;
    --effector)
      EFFECTOR="${2:?missing effector}"; shift 2 ;;
    --namespace)
      NAMESPACE="${2:?missing namespace}"; shift 2 ;;
    --enable)
      AUTO_ENABLE=true; shift ;;
    -h|--help)
      echo 'Usage: start_real.sh --can can0 [--effector none|agx_gripper|revo2|revo2_pro|revo2_touch] [--namespace NAME] [--enable]'
      echo 'Without --enable, the driver connects and reads state but does not automatically enable the arm.'
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CAN_PORT" ]] || {
  echo '--can is required' >&2; exit 2;
}
[[ "$CAN_PORT" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
  echo 'Invalid CAN interface name' >&2; exit 2;
}
case "$EFFECTOR" in
  none|agx_gripper|revo2|revo2_pro|revo2_touch) ;;
  *) echo "Unsupported effector: $EFFECTOR" >&2; exit 2 ;;
esac
ip -o link show "$CAN_PORT" >/dev/null 2>&1 || {
  echo "CAN interface $CAN_PORT does not exist" >&2; exit 1;
}

set +u
source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export LC_NUMERIC=en_US.UTF-8
ARGS=("can_port:=$CAN_PORT" arm_type:=piper "effector_type:=$EFFECTOR"
      follow:=true auto_control_gate:=true "auto_enable:=$AUTO_ENABLE")
if [[ -n "$NAMESPACE" ]]; then
  ARGS+=("namespace:=$NAMESPACE")
fi
exec ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py "${ARGS[@]}"
