#!/usr/bin/env python3
"""Convert a single simulated RGB-D grasp inference to Piper candidates.

Inputs must be camera observations; this program never reads Gazebo truth.
It writes a local planning artifact and does not send ROS commands.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
from piperx_lab.graspnet_adapter import convert_grasps
from piperx_lab.cylinder_grasp import model_tip_transform
from piperx_lab.model import installed_gripper_model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--grasps', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-model-grasps', type=int, default=13)
    args = parser.parse_args()
    if not 1 <= args.max_model_grasps <= 13:
        parser.error('max-model-grasps must be 1..13')
    metadata = json.loads(args.metadata.read_text())
    stamp = metadata['sample_stamp']
    joints = metadata['sample_joint_positions_rad']
    transform = metadata['T_base_camera']
    if (metadata.get('source') != 'gazebo_simulated_rgbd' or
            metadata.get('base_frame') != 'base_link' or
            metadata.get('tf_latest_fallback') is not False or
            not np.isfinite(metadata.get('tf_lag_s', float('nan'))) or
            metadata['tf_lag_s'] > 0.05):
        raise ValueError('Expected simulation camera TF at the image timestamp')
    with np.load(args.frame, allow_pickle=False) as data:
        xyz, mask = np.asarray(data['xyz']), np.asarray(data['mask'], dtype=bool)
        if xyz.ndim != 3 or xyz.shape[2] != 3 or mask.shape != xyz.shape[:2]:
            raise ValueError('Invalid camera point cloud or observed target mask')
        if not np.array_equal(np.asarray(data['stamp']), [stamp['sec'], stamp['nanosec']]):
            raise ValueError('Frame and transform metadata have different timestamps')
        target_points = xyz[mask & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)]
    rows = np.load(args.grasps, allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 17:
        raise ValueError('Expected Nx17 GraspNet predictions')
    rows = rows[np.argsort(-rows[:, 0], kind='stable')]
    candidates, rejected = convert_grasps(rows, transform,
        model_tip_transform(installed_gripper_model_path()), np.eye(4), target_points)
    # Keep all three approach distances per chosen model grasp, up to 39.
    model_ids = list(dict.fromkeys(c['model_id'] for c in candidates))[:args.max_model_grasps]
    candidates = [c for c in candidates if c['model_id'] in model_ids]
    result = {'source': 'graspnet_sim_rgbd', 'frame': 'base_link',
              'sample_stamp': stamp, 'sample_joint_positions_rad': joints,
              'T_base_camera': transform, 'candidate_source': str(args.grasps),
              'frame_source': str(args.frame), 'observed_target_points': len(target_points),
              'model_predictions': len(rows), 'rejected_predictions': rejected,
              'candidates': candidates,
              'limitations': ['Piper gripper and GraspNet model differ; geometry alignment is approximate',
                              'not a contact or motion plan validation']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'model_predictions': len(rows), 'candidate_variants': len(candidates),
                      'rejected_predictions': len(rejected)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
