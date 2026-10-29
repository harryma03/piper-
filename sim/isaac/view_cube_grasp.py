#!/usr/bin/env python3
"""Run the upstream automatic cube stack demo with a close-up viewer."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/IsaacLab_Data_Collection/scripts/tools/record_ik_stack.py"

spec = importlib.util.spec_from_file_location("piper_upstream_ik_stack", SOURCE)
upstream = importlib.util.module_from_spec(spec)
sys.argv[0] = str(SOURCE)
spec.loader.exec_module(upstream)
original_config = upstream.create_environment_config


def viewer_config(*args, **kwargs):
    cfg, success = original_config(*args, **kwargs)
    cfg.viewer.eye = (1.1, 1.0, 0.9)
    cfg.viewer.lookat = (0.4, 0.0, 0.15)
    return cfg, success


upstream.create_environment_config = viewer_config
try:
    upstream.main()
finally:
    upstream.simulation_app.close()
