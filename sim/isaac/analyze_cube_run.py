#!/usr/bin/env python3
"""Check recorded cube trajectories independently of the upstream success flag."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = json.loads((args.run_dir / "result.json").read_text())
    with (args.run_dir / "trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    results = []

    def cube_position(row, n):
        return np.array([float(row[f"cube{n}_{axis}"]) for axis in "xyz"])

    for trial in run["trials"]:
        data = [row for row in rows if int(row["trial"]) == trial["trial"]]
        if not data:
            results.append({"trial": trial["trial"], "status": "no_samples"})
            continue
        last = data[-1]
        positions = [cube_position(last, n) for n in (1, 2, 3)]
        gaps = []
        for low, high in ((0, 1), (1, 2)):
            d = positions[high] - positions[low]
            gaps.append({"lower": low + 1, "upper": high + 1,
                         "height_m": float(d[2]), "xy_offset_m": float(np.linalg.norm(d[:2]))})
        tail = data[-10:]
        travel = [max(float(np.linalg.norm(cube_position(row, n) - cube_position(tail[0], n)))
                      for row in tail) for n in (1, 2, 3)]
        gripper_open = len(tail) == 10 and all(
            float(row["command_grip"]) > 0
            and abs(float(row["position_gripper_joint1"]) - 0.05) < 0.001
            and abs(float(row["position_gripper_joint2"]) + 0.05) < 0.001 for row in tail)
        geometry = all(g["xy_offset_m"] < 0.01 and abs(g["height_m"] - 0.0468) < 0.005 for g in gaps)
        stable_tail = len(tail) == 10 and max(travel) < 0.003
        drift = {}
        for n in (2, 3):
            eligible = [row for row in data if float(row["command_grip"]) < 0
                        and float(row[f"cube{n}_z"]) - trial["initial_z_m"][n - 1] > 0.05]
            groups = []
            for row in eligible:
                if not groups or int(row["step"]) != int(groups[-1][-1]["step"]) + 1:
                    groups.append([])
                groups[-1].append(row)
            if not groups:
                drift[str(n)] = None
                continue
            interval = max(groups, key=len)
            relative = []
            for row in interval:
                ee = np.array([float(row[f"ee_{axis}"]) for axis in "xyz"])
                rotation = Rotation.from_quat([float(row[f"ee_q{axis}"]) for axis in "xyzw"])
                relative.append(rotation.inv().apply(cube_position(row, n) - ee))
            relative = np.asarray(relative)
            drift[str(n)] = {"samples": len(interval), "start_step": int(interval[0]["step"]),
                             "end_step": int(interval[-1]["step"]),
                             "max_displacement_from_first_m": float(np.linalg.norm(relative - relative[0], axis=1).max())}
        results.append({"trial": trial["trial"], "upstream_success": trial["upstream_success"],
                        "stop_reason": trial["stop_reason"], "steps": len(data),
                        "final_gaps": gaps, "gripper_open_last_10": gripper_open,
                        "max_cube_motion_last_10_m": travel, "stable_last_10": stable_tail,
                        "strict_stack_geometry": geometry,
                        "independent_stack_check": (geometry and stable_tail and gripper_open
                                                    and trial["stop_reason"] != "environment_terminated_or_reset"),
                        "closed_lift_relative_motion": drift})
    report = {"run_status": run["status"], "criteria": "XY < 10 mm; vertical gap within 5 mm of 46.8 mm; "
              "gripper open and each cube moves < 3 mm over last 10 samples; no direct contact-force measurement",
              "trials": results}
    path = args.run_dir / "analysis.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
