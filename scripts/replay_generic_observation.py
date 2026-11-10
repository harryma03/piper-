#!/usr/bin/env python3
"""Replay saved trial frames through the generic observation pipeline offline.

For every records/graspnet_gazebo/*/trial_* directory with a frame.npz this
rebuilds the shape-agnostic observation from the recorded RGB-D frame and
compares it against the spawn truth from the batch summary and against the
legacy fixed-shape fit stored beside it. Spawn truth is used only for scoring,
never as an input to the observation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
from piperx_lab.object_observation import build_generic_observation  # noqa: E402

TABLE_HEIGHT_M = -0.15
LEGACY_HEIGHT_M = {'sim_rgbd_red_cylinder_fit': 0.08, 'sim_rgbd_red_cube_fit': 0.046}
LEGACY_DIMS_M = {'sim_rgbd_red_cylinder_fit': [0.046, 0.046, 0.08],
                 'sim_rgbd_red_cube_fit': [0.046, 0.046, 0.046]}


def trial_spawn(summary, trial_dir):
    """Spawn XY truth recorded by the batch runner, or None."""
    results = summary.get('trial_results') or []
    try:
        index = int(trial_dir.name.split('_')[1]) - 1
    except (IndexError, ValueError):
        return None
    if 0 <= index < len(results):
        return (results[index] or {}).get('spawn_xy_m')
    return None


def replay_frame(trial_dir, stride=2):
    frame_path = trial_dir / 'frame.npz'
    metadata_path = trial_dir / 'frame.json'
    legacy_path = trial_dir / 'frame_observation.json'
    if not (frame_path.is_file() and metadata_path.is_file()):
        return {'trial': str(trial_dir), 'status': 'missing_inputs'}
    metadata = json.loads(metadata_path.read_text())
    legacy = json.loads(legacy_path.read_text()) if legacy_path.is_file() else {}
    with np.load(frame_path, allow_pickle=False) as data:
        xyz = np.asarray(data['xyz'], dtype=np.float64)
        mask = np.asarray(data['mask'], dtype=bool)
    T = np.asarray(metadata['T_base_camera'], dtype=float)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
    scene = xyz[valid][::stride]
    mask_points = xyz[mask & valid]
    built = build_generic_observation(mask_points, scene, T,
                                      stamp=metadata.get('sample_stamp'),
                                      camera_frame=metadata.get('camera_frame', ''),
                                      joint_positions_rad=metadata.get('sample_joint_positions_rad'))
    result = {'trial': str(trial_dir), 'status': built['status']}
    if built['status'] != 'ok':
        result['reason'] = built.get('reason')
        return result
    observation = built['observation']
    result['center_base_m'] = observation['obb_center_base_m']
    result['dimensions_m'] = observation['dimensions_m']
    result['table_source'] = observation['quality']['table_source']
    result['table_height_m'] = observation['quality']['table_height_m']
    result['symmetry'] = observation['symmetry']['vertical_axis']['class']
    result['confidence'] = observation['segmentation_confidence']
    result['points_after_filter'] = observation['quality']['points_after_filter']

    source = legacy.get('source')
    if source in LEGACY_HEIGHT_M and legacy.get('center_xyz_m'):
        height = LEGACY_HEIGHT_M[source]
        legacy_center = np.asarray(legacy['center_xyz_m'])
        result['legacy_center_base_m'] = legacy_center.tolist()
        result['legacy_dims_m'] = LEGACY_DIMS_M[source]
    else:
        height = None
    if height is not None:
        result['expected_height_m'] = height
    return result


def score_against_truth(result, spawn_xy, height):
    """Add truth-based errors; truth feeds only this scoring step."""
    if spawn_xy is None or height is None or result['status'] != 'ok':
        return
    center = np.asarray(result['center_base_m'])
    truth = np.array([spawn_xy[0], spawn_xy[1], TABLE_HEIGHT_M + height / 2])
    result['truth_center_base_m'] = truth.tolist()
    result['center_error_m'] = float(np.linalg.norm(center - truth))
    legacy = result.get('legacy_center_base_m')
    if legacy is not None:
        result['legacy_center_error_m'] = float(
            np.linalg.norm(np.asarray(legacy) - truth))
    dims = np.asarray(result['dimensions_m'])
    legacy_dims = result.get('legacy_dims_m')
    if legacy_dims is not None:
        result['dims_error_m'] = [float(v) for v in np.abs(dims - np.asarray(legacy_dims))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-root', type=Path, default=ROOT / 'records/graspnet_gazebo')
    parser.add_argument('--out', type=Path, default=ROOT / 'tmp/generic_observation_replay.json')
    parser.add_argument('--stride', type=int, default=2)
    args = parser.parse_args()
    results = []
    for summary_path in sorted(args.records_root.glob('*/summary.json')):
        summary = json.loads(summary_path.read_text())
        for trial_dir in sorted(summary_path.parent.glob('trial_*')):
            if not (trial_dir / 'frame.npz').is_file():
                continue
            result = replay_frame(trial_dir, stride=args.stride)
            legacy = json.loads((trial_dir / 'frame_observation.json').read_text()) \
                if (trial_dir / 'frame_observation.json').is_file() else {}
            height = LEGACY_HEIGHT_M.get(legacy.get('source'))
            score_against_truth(result, trial_spawn(summary, trial_dir), height)
            results.append(result)
            if result['status'] == 'ok':
                print(f"{result['trial']}: center_err="
                      f"{result.get('center_error_m', float('nan')) * 1e3:.1f} mm "
                      f"(legacy {result.get('legacy_center_error_m', float('nan')) * 1e3:.1f} mm) "
                      f"dims={[round(v * 1e3, 1) for v in result['dimensions_m']]} mm "
                      f"table={result['table_source']}", flush=True)
            else:
                print(f"{result['trial']}: {result['status']}: {result.get('reason')}", flush=True)
    ok = [r for r in results if r['status'] == 'ok']
    scored = [r for r in ok if 'center_error_m' in r]
    summary_out = {
        'frames': len(results),
        'ok': len(ok),
        'rejected': [ {'trial': r['trial'], 'reason': r.get('reason')}
                      for r in results if r['status'] != 'ok'],
        'scored_against_truth': len(scored),
        'center_error_mm': {
            'max': max((r['center_error_m'] for r in scored), default=None) and
                   max(r['center_error_m'] for r in scored) * 1e3,
            'mean': float(np.mean([r['center_error_m'] for r in scored])) * 1e3 if scored else None,
            'under_5mm': sum(r['center_error_m'] < 0.005 for r in scored),
        },
        'legacy_center_error_mm': {
            'max': max((r['legacy_center_error_m'] for r in scored if 'legacy_center_error_m' in r),
                       default=None) and
                   max(r['legacy_center_error_m'] for r in scored if 'legacy_center_error_m' in r) * 1e3,
            'mean': float(np.mean([r['legacy_center_error_m'] for r in scored
                                   if 'legacy_center_error_m' in r])) * 1e3
            if any('legacy_center_error_m' in r for r in scored) else None,
        },
        'dims_max_error_mm': {
            trial: [round(v * 1e3, 2) for v in trial_result['dims_error_m']]
            for trial, trial_result in ((r['trial'], r) for r in scored)
            if 'dims_error_m' in trial_result
        },
        'results': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary_out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary_out.items() if k != 'results'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
