#!/usr/bin/env python3
"""Annotate merged GraspNet candidates against the observation cloud.

Every candidate is checked with the shared revalidation (local width
feasibility, jaw containment, bilateral contact support). Valid candidates
gain a heuristic confidence for ordering; invalid ones keep their rejection
reason so the batch can report why they died. Nothing is silently dropped.
Legacy observations without a cloud file pass through unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
from piperx_lab.grasp_candidates import apply_revalidation  # noqa: E402
from piperx_lab.object_observation import (  # noqa: E402
    SOURCE as GENERIC_SOURCE,
    read_ply,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.candidates.read_text())
    obs = json.loads(args.observation.read_text())
    if (data.get('source') not in ('graspnet_sim_rgbd', 'graspnet_fusion')
            or data.get('frame') != 'base_link'
            or obs.get('status') != 'ok'
            or obs.get('base_frame') != 'base_link'
            or obs.get('stamp') != data.get('sample_stamp')):
        raise ValueError('Expected matching simulated RGB-D candidates and observation')
    candidates = data['candidates']
    if not candidates:
        raise ValueError('No input candidates')
    cloud = None
    if obs.get('source') == GENERIC_SOURCE:
        relative = obs.get('points_base_path')
        if not relative:
            raise ValueError('Generic observation carries no points_base_path')
        with np.load(args.observation.parent / relative, allow_pickle=False) as npz:
            cloud = np.asarray(npz['points_base'], dtype=float)
        hull_vertices = None
        if obs.get('convex_hull_path'):
            hull_vertices, _ = read_ply(args.observation.parent / obs['convex_hull_path'])
        table_height = (obs.get('quality') or {}).get('table_height_m')
        candidates = apply_revalidation(candidates, cloud, hull_vertices=hull_vertices,
                                        table_height_m=table_height)
    data['candidates'] = candidates
    data['candidates_revalidated'] = cloud is not None
    data['rejection_summary'] = {}
    if cloud is not None:
        for candidate in candidates:
            reason = candidate.get('rejection_reason')
            if reason:
                data['rejection_summary'][reason] = data['rejection_summary'].get(reason, 0) + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    valid = sum(1 for c in candidates if not c.get('rejection_reason'))
    print(json.dumps({'candidates': len(candidates), 'valid': valid,
                      'revalidated': cloud is not None,
                      'rejections': data['rejection_summary']}))


if __name__ == '__main__':
    main()
