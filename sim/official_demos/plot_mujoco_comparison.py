#!/usr/bin/env python3
"""Plot recorded free-cube diagnostics; does not run the simulator."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "records/official_demo_tests"
CASES = [
    ("baseline", "Kp 40, friction 1.0", "#c43d3d", "-"),
    ("kp80_mu1", "Kp 80, friction 1.0", "#167a58", "-"),
    ("kp40_mu2", "Kp 40, friction 2.0", "#2e65ad", "--"),
]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
for key, label, color, style in CASES:
    with (RECORDS / f"mujoco_cube_{key}/trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = [float(r["time_s"]) for r in rows]
    lift_mm = [1000 * float(r["cube_lift_m"]) for r in rows]
    force = [float(r["finger1_normal_sum_n"]) + float(r["finger2_normal_sum_n"]) for r in rows]
    axes[0].plot(times, lift_mm, label=label, color=color, linestyle=style, linewidth=2)
    axes[1].plot(times, force, label=label, color=color, linestyle=style, linewidth=1.5)
for ax in axes:
    ax.axvspan(0.5, 2.5, color="#a7b6c8", alpha=.10)
    ax.axvspan(2.5, 5.5, color="#92c8ad", alpha=.12)
    ax.axvspan(5.5, 6.5, color="#edc58a", alpha=.15)
    ax.grid(alpha=.25)
    ax.set_xlim(0, 6.5)
axes[0].axhline(50, color="#555555", linestyle="--", linewidth=1)
axes[0].set_ylabel("Cube lift (mm)")
axes[0].legend(loc="upper left")
axes[1].set_ylabel("Sum of finger normal forces (N)")
axes[1].set_xlabel("Simulation time (s)   |   close 0.5–2.5   |   lift 2.5–5.5   |   hold 5.5–6.5")
fig.suptitle("Piper MuJoCo: free cube grasp comparison\n40 mm, 80 g cube starts between the fingers; no attachment", fontsize=13)
out = RECORDS / "mujoco_cube_comparison.png"
fig.savefig(out, dpi=150)
print(out)
