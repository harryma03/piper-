#!/usr/bin/env python3
"""Rotate grasp candidates about the observed vertical target axis -- then
re-validate every rotated candidate against the observation cloud.

For a generic observation the rotation centre is the observed supported box
centre and the sampling follows the observation's symmetry hypothesis:
'likely' rotates evenly as a circular-symmetric hypothesis, 'unlikely' uses
the discrete orders recorded there, 'unknown' keeps only the unrotated
candidates unless --force. Every rotated candidate passes the same shared
revalidation (width feasibility, jaw containment, bilateral support) against
the base-frame cloud; failures are dropped and their reasons recorded.
Rotated candidates that survive carry ``rotated_target_contact_revalidated:
true``; legacy cylinder/cube observations have no cloud file and keep the old
unrevalidated behaviour.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
from piperx_lab.grasp_candidates import apply_revalidation  # noqa: E402
from piperx_lab.object_observation import (  # noqa: E402
    SOURCE as GENERIC_SOURCE,
    GenericObjectObservation,
    read_ply,
)


def load_observation_cloud(observation_path, observation):
    """Base-frame cloud and hull vertices from a generic observation, or (None, None)."""
    if observation.get('source') != GENERIC_SOURCE:
        return None, None
    relative = observation.get('points_base_path')
    if not relative:
        raise ValueError('Generic observation carries no points_base_path')
    points_path = Path(observation_path).parent / relative
    with np.load(points_path, allow_pickle=False) as data:
        points = np.asarray(data['points_base'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Invalid base-frame cloud in {points_path}')
    hull_vertices = None
    if observation.get('convex_hull_path'):
        hull_vertices, _ = read_ply(Path(observation_path).parent / observation['convex_hull_path'])
    return points, hull_vertices


def generic_rotation_plan(observation, samples, force):
    """Angles and rotation centre for a generic observation."""
    parsed = GenericObjectObservation.from_dict(observation)
    center = np.asarray(parsed.center(), dtype=float)
    hypothesis = parsed.vertical_axis_symmetry()
    orders = (parsed.symmetry.get('vertical_axis') or {}).get('discrete_orders_deg')
    if hypothesis == 'likely':
        return center, [2 * math.pi * i / samples for i in range(samples)], \
            'vertical-axis rotational symmetry hypothesis from the generic observation'
    if hypothesis == 'unlikely' and orders:
        return center, [math.radians(a) for a in orders], \
            f'discrete symmetry orders {orders} from the generic observation'
    if force:
        return center, [2 * math.pi * i / samples for i in range(samples)], \
            f'forced even sampling on an {hypothesis}-symmetry observation'
    # Unknown symmetry keeps the unrotated model candidates; no symmetry is
    # assumed and no candidate set is lost.
    return center, [0.0], \
        f'unclear vertical-axis symmetry ({hypothesis}); unrotated candidates only'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True, type=Path)
    p.add_argument('--observation', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--top-models', type=int, default=1,
                   help='Rotate this many distinct top-scored model candidates')
    p.add_argument('--object-shape', choices=('cylinder', 'cube'), default=None,
                   help='Legacy fixed-shape regression paths only')
    p.add_argument('--force', action='store_true',
                   help='Rotate even when the observation reports unclear symmetry')
    args = p.parse_args()
    if not 1 <= args.samples <= 20:
        p.error('samples must be in 1..20')
    if not 1 <= args.top_models <= 4 or args.samples * args.top_models > 40:
        p.error('top-models must be 1..4 and total candidates must not exceed 40')
    data = json.loads(args.candidates.read_text())
    obs = json.loads(args.observation.read_text())
    if (data.get('source') not in ('graspnet_sim_rgbd', 'graspnet_fusion')
            or data.get('frame') != 'base_link'
            or obs.get('status') != 'ok'
            or obs.get('base_frame') != 'base_link'
            or obs.get('stamp') != data.get('sample_stamp')):
        raise ValueError('Expected matching simulated RGB-D target observation and grasp candidates')
    cloud, hull_vertices = load_observation_cloud(args.observation, obs)
    if obs.get('source') == GENERIC_SOURCE:
        center, angles, assumption = generic_rotation_plan(obs, args.samples, args.force)
        shape_label = f"generic/{(obs.get('symmetry', {}).get('vertical_axis') or {}).get('class')}"
    else:
        expected_source = {'cylinder': 'sim_rgbd_red_cylinder_fit',
                           'cube': 'sim_rgbd_red_cube_fit'}.get(args.object_shape)
        if (expected_source is None or obs.get('source') != expected_source
                or obs.get('shape', args.object_shape) != args.object_shape):
            raise ValueError('Expected matching simulated RGB-D target observation and grasp candidates')
        center = np.asarray(obs['center_xyz_m'], dtype=float)
        angles = [2 * math.pi * i / args.samples for i in range(args.samples)]
        assumption = ('upright circular cylinder; observed center and dimensions use task priors'
                      if args.object_shape == 'cylinder' else
                      'upright square cube; observed center and dimensions use task priors')
        shape_label = args.object_shape
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Invalid observed center')
    candidates = data['candidates']
    if not candidates:
        raise ValueError('No input candidates')
    if len(angles) * args.top_models > 40:
        p.error('total symmetry candidates must not exceed 40')
    # Several approach distances and input crops can refer to the same model
    # grasp.  Keep its shortest pregrasp offset, then rotate the best distinct
    # model grasps about the observed vertical axis.
    ranked = sorted(candidates, key=lambda c: (-(c.get('confidence') or c.get('score') or 0),
                                               c.get('approach_distance_m', 1.0)))
    selected, seen = [], set()
    for candidate in ranked:
        key = (candidate.get('source_roi_padding_px'), candidate.get('model_id'),
               candidate.get('raw_azimuth_deg'), candidate.get('raw_tilt_deg'))
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) == args.top_models:
            break
    result, candidate_id = [], 0
    rotation_rejections = []
    for model_rank, source in enumerate(selected):
        surviving = 0
        for angle in angles:
            c, s = math.cos(angle), math.sin(angle)
            T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = center - T[:3, :3] @ center
            item = copy.deepcopy(source)
            for key in ('T_base_tip', 'T_base_tcp', 'T_base_pregrasp_tcp'):
                item[key] = (T @ np.asarray(item[key])).tolist()
            item['approach_direction'] = (
                T[:3, :3] @ np.asarray(item['approach_direction'])).tolist()
            item['symmetry_angle_deg'] = float(math.degrees(angle))
            item['cylinder_symmetry_angle_deg'] = float(math.degrees(angle))
            item['source_model_id'] = source.get('model_id')
            item['symmetry_model_rank'] = model_rank
            item['target_points_in_jaw_volume'] = None
            if cloud is not None:
                # Rotated candidates are hypotheses until re-checked against
                # the observed cloud; failures are data, not silent drops.
                checked = apply_revalidation(
                    [item], cloud, hull_vertices=hull_vertices,
                    table_height_m=(obs.get('quality') or {}).get('table_height_m'))
                item = checked[0]
                if item['rejection_reason'] is not None:
                    rotation_rejections.append({
                        'model_rank': model_rank,
                        'angle_deg': float(math.degrees(angle)),
                        'reason': item['rejection_reason'],
                    })
                    continue
                item['rotated_target_contact_revalidated'] = True
                item['id'] = candidate_id
                candidate_id += 1
                result.append(item)
                surviving += 1
            else:
                # Legacy fixed-shape regression path: no cloud to check against.
                item['rotated_target_contact_revalidated'] = False
                item['id'] = candidate_id
                candidate_id += 1
                result.append(item)
                surviving += 1
        if surviving == 0:
            rotation_rejections.append({'model_rank': model_rank,
                                        'reason': 'all_rotations_rejected'})
    if not result:
        # Every rotation failed revalidation (or the payload was already
        # empty of valid candidates): emit an empty candidate list so the
        # executor records a planning-stage failure instead of the batch
        # aborting here. The rejection reasons above remain the evidence.
        data['candidates'] = []
    else:
        data['candidates'] = result
    data['symmetry_assumption'] = assumption
    data['symmetry_object_shape'] = shape_label
    data['symmetry_samples'] = len(angles)
    data['symmetry_top_models'] = len(selected)
    data['rotated_candidates_revalidated'] = cloud is not None
    data['rotation_rejections'] = rotation_rejections
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'candidates': len(result), 'revalidation_rejections': len(rotation_rejections),
                      'models': [
        {'model_id': item.get('model_id'),
         'source_roi_padding_px': item.get('source_roi_padding_px'),
         'score': item.get('score'), 'confidence': item.get('confidence')} for item in selected]}))


if __name__ == '__main__':
    main()
