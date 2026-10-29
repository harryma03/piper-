#!/usr/bin/env python3
"""Plot recorded Isaac Lab cube heights and simulated gripper opening."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
parser.add_argument("--trial", type=int, default=0)
args = parser.parse_args()
with (args.run_dir / "trajectory.csv").open(newline="") as stream:
    rows = [r for r in csv.DictReader(stream) if int(r["trial"]) == args.trial]
if not rows:
    raise SystemExit("No samples for requested trial")
times = [float(r["sim_time_s"]) for r in rows]
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
for n, color in ((1, "#2e65ad"), (2, "#c43d3d"), (3, "#167a58")):
    axes[0].plot(times, [1000 * float(r[f"cube{n}_z"]) for r in rows], label=f"Cube {n}", color=color, linewidth=2)
axes[0].set_ylabel("Cube center height (mm)")
axes[0].legend(loc="upper left")
opening = [1000 * abs(float(r["position_gripper_joint1"]) - float(r["position_gripper_joint2"])) for r in rows]
axes[1].plot(times, opening, color="#6245a0", linewidth=2)
axes[1].set_ylabel("Difference in finger joint positions (mm)")
axes[1].set_xlabel("Simulation time (s)")
for ax in axes:
    ax.grid(alpha=0.25)
    ax.set_xlim(times[0], times[-1])
fig.suptitle("Isaac Lab Piper: original cube stack controller\nSimulated ground-truth positions; no camera input", fontsize=13)
out = args.run_dir / f"trajectory_trial_{args.trial}.png"
fig.savefig(out, dpi=150)
print(out)
