#!/usr/bin/env python3
"""Generate the fixed 50-scene grasp benchmark suite.

The suite is deterministic: the same seed always yields the same scenes, and
the scene list is written to JSON for both the "before" and "after" runs so
the comparison is on identical scenes. Scene parameters feed the simulator
(truth) and the independent evaluator only -- perception never reads them.
"""
import argparse
import json
import math
from pathlib import Path
import random

CENTER = (0.45, 0.0)


def scene(shape, dims, xy, yaw=0.0, occluder=False, view=0, group=""):
    return {"shape": shape, "dims": dims, "xy": [round(xy[0], 4), round(xy[1], 4)],
            "yaw": round(yaw, 2), "occluder": occluder, "view": view,
            "group": group}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    scenes = []

    def jitter(radius=0.05):
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.0, radius)
        return (CENTER[0] + r * math.cos(angle), CENTER[1] + r * math.sin(angle))

    # G1 (6): baseline 46x80 cylinder, jittered near centre -- the historical
    # regression target under the new pipeline.
    for _ in range(6):
        scenes.append(scene('cylinder', [0.046, 0.08], jitter(), group='baseline_cylinder'))

    # G2 (8): small targets below the old fixed 32 mm close width.
    small = [
        ('cylinder', [0.024, 0.07]), ('cylinder', [0.028, 0.06]),
        ('cube', [0.024, 0.024, 0.024]), ('box', [0.030, 0.022, 0.040]),
        ('sphere', [0.028]), ('cylinder', [0.030, 0.055]),
        ('box', [0.026, 0.026, 0.050]), ('sphere', [0.024]),
    ]
    for shape, dims in small:
        scenes.append(scene(shape, dims, jitter(), group='small'))

    # G3 (8): large/wide targets near the 70 mm opening limit.
    large = [
        ('cylinder', [0.060, 0.060]), ('cylinder', [0.057, 0.053]),
        ('box', [0.060, 0.060, 0.040]), ('box', [0.065, 0.028, 0.045]),
        ('bar', [0.055, 0.022, 0.070]), ('cylinder', [0.050, 0.090]),
        ('box', [0.049, 0.055, 0.032]), ('bar', [0.060, 0.024, 0.024]),
    ]
    for shape, dims in large:
        scenes.append(scene(shape, dims, jitter(), group='large'))

    # G4 (8): orientation coverage on boxes and rectangular blocks.
    oriented = [
        ('cube', [0.046, 0.046, 0.046], 30.0), ('cube', [0.046, 0.046, 0.046], 90.0),
        ('cube', [0.046, 0.046, 0.046], 180.0), ('box', [0.040, 0.030, 0.030], 45.0),
        ('box', [0.055, 0.040, 0.028], 60.0), ('box', [0.048, 0.048, 0.036], 15.0),
        ('bar', [0.080, 0.020, 0.020], 90.0), ('box', [0.036, 0.036, 0.060], 120.0),
    ]
    for shape, dims, yaw in oriented:
        scenes.append(scene(shape, dims, jitter(), yaw=yaw, group='oriented'))

    # G5 (4): spheres.
    for dims in ([0.040], [0.034], [0.046], [0.026]):
        scenes.append(scene('sphere', dims, jitter(), group='sphere'))

    # G6 (6): partial occlusion by a static white blocker.
    occluded = [
        ('cylinder', [0.046, 0.08]), ('cube', [0.046, 0.046, 0.046]),
        ('box', [0.040, 0.030, 0.040]), ('cylinder', [0.030, 0.06]),
        ('sphere', [0.040]), ('box', [0.055, 0.030, 0.035]),
    ]
    for shape, dims in occluded:
        scenes.append(scene(shape, dims, jitter(), occluder=True, group='occluded'))

    # G7 (6): workspace edges, where reachability is hardest.
    edges = [(0.40, 0.0), (0.50, 0.0), (0.45, 0.05), (0.45, -0.05),
             (0.50, -0.04), (0.40, 0.045)]
    edge_objects = [('cylinder', [0.046, 0.08]), ('cube', [0.046, 0.046, 0.046]),
                    ('box', [0.040, 0.030, 0.040]), ('sphere', [0.040]),
                    ('cylinder', [0.032, 0.06]), ('box', [0.050, 0.028, 0.036])]
    for (xy, (shape, dims)) in zip(edges, edge_objects):
        scenes.append(scene(shape, dims, xy, group='edge'))

    # G8 (4): camera view variants (base-yaw azimuth shifts).
    views = [
        ('cylinder', [0.046, 0.08], 1), ('cube', [0.046, 0.046, 0.046], 2),
        ('box', [0.045, 0.032, 0.042], 1), ('sphere', [0.038], 2),
    ]
    for shape, dims, view in views:
        scenes.append(scene(shape, dims, jitter(), view=view, group='view'))

    assert len(scenes) == 50, f'expected 50 scenes, got {len(scenes)}'
    payload = {
        'seed': args.seed,
        'count': len(scenes),
        'success_definitions': {
            'physical_grasp_success': ('both finger contacts in closing and lifting AND '
                                       'lift > 0.07 m AND xy offset < 0.06 m'),
            'success': 'physical_grasp_success AND hold-stability gate (mode per observation)',
            'motion_feedback_settled': 'lift joint feedback reached command within 0.015',
            'execution_complete': 'success AND motion_feedback_settled; primary endpoint',
        },
        'notes': [
            'scene truth feeds the simulator and the evaluator only',
            'perception, candidates, ranking and execution use camera data and robot state alone',
            'infra failures (gazebo/cuda/capture) are reported separately, excluded from denominators',
        ],
        'scenes': scenes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    groups = {}
    for item in scenes:
        groups[item['group']] = groups.get(item['group'], 0) + 1
    print(json.dumps({'out': str(args.out), 'count': len(scenes), 'groups': groups}))


if __name__ == '__main__':
    main()
