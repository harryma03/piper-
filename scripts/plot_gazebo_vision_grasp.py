#!/usr/bin/env python3
"""Plot completed random grasp trials from a saved local summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    trials = report['trial_results']
    fig, ax = plt.subplots(figsize=(7.2, 5.7), constrained_layout=True)
    for success, label, color, marker in [(True, 'Passed grasp checks', '#27834c', 'o'),
                                          (False, 'Failed / incomplete', '#b83232', 'X')]:
        data = [t for t in trials if t.get('success') is success]
        if data:
            ax.scatter([t['spawn_xy_m'][0]*100 for t in data],
                       [t['spawn_xy_m'][1]*100 for t in data],
                       color=color, marker=marker, s=85, label=label, zorder=3)
            for t in data:
                ax.annotate(str(t['trial']), (t['spawn_xy_m'][0]*100, t['spawn_xy_m'][1]*100),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
    region = report['region_m']
    ax.set_xlim(region['x'][0]*100-1, region['x'][1]*100+1)
    ax.set_ylim(region['y'][0]*100-1, region['y'][1]*100+1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Base X (cm)')
    ax.set_ylabel('Base Y (cm)')
    ax.set_title(f"Simulated RGB-D grasp: {report['success_count']}/{len(trials)} passed")
    ax.grid(alpha=.25)
    ax.legend(loc='upper left')
    output = args.out or args.summary.with_name('positions.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)

if __name__ == '__main__':
    main()
