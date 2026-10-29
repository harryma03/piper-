#!/usr/bin/env bash
# Single-environment simulation only. No ROS or CAN connection is started.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PYTHON="$PROJECT_ROOT/.isaac-venv/bin/python"
ISAAC_RUN_DIR="${ISAAC_RUN_DIR:-$PROJECT_ROOT/records/isaac_stack/$(date +%Y%m%d_%H%M%S)}"
ISAAC_TIMEOUT_SECONDS="${ISAAC_TIMEOUT_SECONDS:-600}"
mkdir -p "$ISAAC_RUN_DIR"
if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo "Isaac Python environment is missing: $ISAAC_PYTHON" >&2
    exit 3
fi
"$ISAAC_PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys
missing = []
for name in ("isaacsim", "isaaclab", "isaaclab-assets", "isaaclab-tasks", "isaaclab-mimic", "agx-teleop"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        missing.append(name)
if missing:
    print("Dependencies are not installed: " + ", ".join(missing), file=sys.stderr)
    sys.exit(3)
PY
# Scope the memory limit to this process tree; no system setting is changed.
SCOPE=()
if command -v systemd-run >/dev/null && systemctl --user is-system-running --quiet; then
    SCOPE=(systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0)
fi
echo "Simulation log: $ISAAC_RUN_DIR/run.log"
"${SCOPE[@]}" timeout --signal=INT --kill-after=30s "${ISAAC_TIMEOUT_SECONDS}s" \
    env -u PYTHONPATH -u LD_LIBRARY_PATH OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    "$ISAAC_PYTHON" "$PROJECT_ROOT/sim/isaac/run_cube_grasp.py" \
    --headless --device cuda --trials 1 --max-steps 700 --output "$ISAAC_RUN_DIR" "$@" \
    2>&1 | tee "$ISAAC_RUN_DIR/run.log"
