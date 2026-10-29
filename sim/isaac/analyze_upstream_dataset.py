#!/usr/bin/env python3
"""Independently inspect unmodified upstream demonstration HDF5 recordings."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path)
args = parser.parse_args()
report = {"dataset": str(args.dataset.resolve()), "episodes": []}
with h5py.File(args.dataset, "r") as source:
    for name, episode in source["data"].items():
        if not isinstance(episode, h5py.Group):
            continue
        positions = np.stack([episode[f"states/rigid_object/cube_{i}/root_pose"][:, :3]
                              for i in (1, 2, 3)], axis=1)
        initial = np.stack([episode[f"initial_state/rigid_object/cube_{i}/root_pose"][0, :3]
                            for i in (1, 2, 3)])
        actions = episode["actions"][:]
        gripper = episode["obs/gripper_pos"][:]
        eef_pos = episode["obs/eef_pos"][:]
        eef_quat_wxyz = episode["obs/eef_quat"][:]
        # The recorder stores policy observations before each step and rigid
        # states after it. Align cube position to observation time explicitly.
        positions_before_step = np.concatenate((initial[None, :, :], positions[:-1]), axis=0)
        gaps = positions[-1, 1:] - positions[-1, :-1]
        xy = np.linalg.norm(gaps[:, :2], axis=1)
        last_motion = np.linalg.norm(positions[-10:] - positions[-10], axis=2).max(axis=0)
        open_last_10 = len(gripper) >= 10 and bool(np.all(np.abs(gripper[-10:] - 0.05) < 0.001))
        geometry = bool(np.all(xy < 0.01) and np.all(np.abs(gaps[:, 2] - 0.0468) < 0.005))
        stable = bool(len(positions) >= 10 and np.all(last_motion < 0.003))
        drift = {}
        for cube in (1, 2):
            indices = np.where((actions[:, -1] < 0)
                               & (positions_before_step[:, cube, 2] - initial[cube, 2] > 0.05))[0]
            sequences = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
            sequence = max(sequences, key=len, default=np.array([], dtype=int))
            if len(sequence) == 0:
                drift[str(cube + 1)] = None
                continue
            q = eef_quat_wxyz[sequence][:, [1, 2, 3, 0]]
            relative = Rotation.from_quat(q).inv().apply(positions_before_step[sequence, cube] - eef_pos[sequence])
            drift[str(cube + 1)] = {"samples": len(sequence),
                "max_displacement_from_first_m": float(np.linalg.norm(relative - relative[0], axis=1).max())}
        report["episodes"].append({"name": name, "steps": len(actions),
            "max_lift_m": (positions[:, :, 2].max(axis=0) - initial[:, 2]).tolist(),
            "height_gaps_m": gaps[:, 2].tolist(), "xy_offsets_m": xy.tolist(),
            "max_cube_motion_last_10_m": last_motion.tolist(), "gripper_open_last_10": open_last_10,
            "independent_stack_check": geometry and stable and open_last_10,
            "closed_lift_relative_motion": drift})
output = args.dataset.with_name("independent_analysis.json")
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
