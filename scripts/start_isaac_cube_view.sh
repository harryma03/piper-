#!/usr/bin/env bash
# Open the simulated Piper cube-stacking demo. No real-robot connection is used.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.isaac-venv/bin/python"
RUN_DIR="${ISAAC_VIEW_DIR:-$PROJECT_ROOT/records/isaac_stack/gui_$(date +%Y%m%d_%H%M%S)}"
RUN_SECONDS="${ISAAC_VIEW_SECONDS:-1800}"
if [[ ! -x "$PYTHON" || -z "${DISPLAY:-}" ]]; then
    echo "Isaac Python environment or graphical display is unavailable." >&2
    exit 1
fi
if systemctl --user is-active --quiet piper-isaac-view.scope; then
    echo "The Piper simulation window is already running."
    exit 0
fi
mkdir -p "$RUN_DIR"
echo "Simulation log: $RUN_DIR/run.log"
cd "$PROJECT_ROOT/third_party/IsaacLab_Data_Collection"
exec systemd-run --user --scope --quiet --unit=piper-isaac-view \
    -p MemoryMax=8G -p MemorySwapMax=0 \
    timeout --signal=INT --kill-after=30s "${RUN_SECONDS}s" \
    env -u PYTHONPATH -u LD_LIBRARY_PATH OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    "$PYTHON" "$PROJECT_ROOT/sim/isaac/view_cube_grasp.py" \
    --task Isaac-Stack-Cube-Piper-IK-Rel-v0 --device cuda --rendering_mode performance \
    --dataset_file "$RUN_DIR/dataset.hdf5" --num_demos 0 --step_hz 20 \
    > "$RUN_DIR/run.log" 2>&1
