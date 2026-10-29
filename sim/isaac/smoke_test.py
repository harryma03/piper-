#!/usr/bin/env python3
"""Run the official create_empty tutorial for 20 physics steps."""
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/IsaacLab/scripts/tutorials/00_sim/create_empty.py"
OUTPUT = ROOT / "records/isaac_stack/smoke_result.json"
result = {"status": "starting", "source": str(SOURCE), "physics_steps": 0}
module = None
try:
    sys.argv[0] = str(SOURCE)
    spec = importlib.util.spec_from_file_location("official_isaac_empty", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_is_running = module.simulation_app.is_running

    def bounded_is_running():
        if result["physics_steps"] >= 20 or not original_is_running():
            return False
        result["physics_steps"] += 1
        return True

    module.simulation_app.is_running = bounded_is_running
    module.main()
    result["status"] = "passed" if result["physics_steps"] == 20 else "stopped_early"
except BaseException as exc:
    result["status"] = "error"
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()
    raise
finally:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if module is not None and hasattr(module, "simulation_app"):
        module.simulation_app.close()
