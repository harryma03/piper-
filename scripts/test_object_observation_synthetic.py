#!/usr/bin/env python3
"""Offline synthetic tests for the generic object observation pipeline.

Builds wrist-camera point clouds of primitive objects on a support plane,
runs the shape-agnostic segmentation, and checks the reconstructed geometry
against the generating truth. No ROS, no Gazebo, no recorded data.
"""
from pathlib import Path
import json
import math
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
from piperx_lab.object_observation import (  # noqa: E402
    GenericObjectObservation,
    build_generic_observation,
    load_object_points,
    read_ply,
    save_generic_observation,
    write_ply,
)
from piperx_lab.grasp_candidate_filter import (  # noqa: E402
    candidate_center_near_target,
    neighborhood_radius_m,
    points_in_jaw_volume,
    validate_candidates_payload,
)

TABLE_Z = -0.15
T_BASE_CAMERA = np.array([
    [1.0, 0.0, 0.0, 0.45],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.2],
    [0.0, 0.0, 0.0, 1.0],
])  # wrist camera above the workspace looking straight down
FAILURES = []


def check(name, condition, detail=''):
    status = 'ok' if condition else 'FAIL'
    print(f'[{status}] {name} {detail}', flush=True)
    if not condition:
        FAILURES.append(name)


def table_points(x_span=(0.35, 0.55), y_span=(-0.10, 0.10), step=0.002):
    xs = np.arange(*x_span, step)
    ys = np.arange(*y_span, step)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return np.column_stack((grid, np.full(len(grid), TABLE_Z)))


def to_camera(points_base):
    """Apply the inverse of T_base_camera (base to camera optical frame)."""
    return (points_base - T_BASE_CAMERA[:3, 3]) @ T_BASE_CAMERA[:3, :3]


def cylinder_cloud(diameter, height, center_xy=(0.45, 0.0)):
    """Top cap plus upper side band, as a wrist camera would see them."""
    radius = diameter / 2
    top_z = TABLE_Z + height
    thetas = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    cap_r = np.linspace(0, radius, 20)
    t, r = np.meshgrid(thetas, cap_r)
    cap = np.column_stack((center_xy[0] + r.ravel() * np.cos(t.ravel()),
                           center_xy[1] + r.ravel() * np.sin(t.ravel()),
                           np.full(t.size, top_z)))
    zs = np.linspace(top_z - 0.05, top_z - 0.002, 40)
    t, z = np.meshgrid(thetas, zs)
    side = np.column_stack((center_xy[0] + radius * np.cos(t.ravel()),
                            center_xy[1] + radius * np.sin(t.ravel()),
                            z.ravel()))
    return cap, side  # cap is white, side band is red


def box_cloud(dims, center_xy=(0.45, 0.0)):
    """Top face plus the upper side band, as a wrist camera would see them."""
    dx, dy, dz = dims
    top_z = TABLE_Z + dz
    xs = np.linspace(center_xy[0] - dx / 2, center_xy[0] + dx / 2, 24)
    ys = np.linspace(center_xy[1] - dy / 2, center_xy[1] + dy / 2, 24)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    top = np.column_stack((grid, np.full(len(grid), top_z)))
    zs = np.linspace(top_z - 0.04, top_z - 0.002, 24)
    side = []
    for z in zs:
        edges_x = np.column_stack((
            np.concatenate((np.full(24, center_xy[0] - dx / 2), np.full(24, center_xy[0] + dx / 2))),
            np.tile(ys, 2)))
        edges_y = np.column_stack((
            np.tile(xs, 2),
            np.concatenate((np.full(24, center_xy[1] - dy / 2), np.full(24, center_xy[1] + dy / 2)))))
        side.append(np.column_stack((np.vstack((edges_x, edges_y)), np.full(96, z))))
    return top, np.vstack(side)


def full_box_cloud(dims, center_xy=(0.45, 0.0)):
    """Complete box surface from support plane to top."""
    dx, dy, dz = dims
    xs = np.linspace(center_xy[0] - dx / 2, center_xy[0] + dx / 2, 20)
    ys = np.linspace(center_xy[1] - dy / 2, center_xy[1] + dy / 2, 20)
    zs = np.linspace(TABLE_Z + 0.002, TABLE_Z + dz, 60)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    top = np.column_stack((grid, np.full(len(grid), TABLE_Z + dz)))
    side = [np.column_stack((
        np.vstack((np.column_stack((np.full(20, center_xy[0] - dx / 2), ys)),
                   np.column_stack((np.full(20, center_xy[0] + dx / 2), ys)),
                   np.column_stack((xs, np.full(20, center_xy[1] - dy / 2))),
                   np.column_stack((xs, np.full(20, center_xy[1] + dy / 2))))),
        np.full(80, z))) for z in zs]
    return np.vstack([top] + side)


def sphere_cloud(diameter, center_xy=(0.45, 0.0)):
    """Upper hemisphere down to the equator resting on the support plane."""
    radius = diameter / 2
    center_z = TABLE_Z + radius
    thetas = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    phis = np.linspace(0, math.pi / 2 - 0.02, 30)
    t, p = np.meshgrid(thetas, phis)
    surface = np.column_stack((
        center_xy[0] + radius * np.sin(p.ravel()) * np.cos(t.ravel()),
        center_xy[1] + radius * np.sin(p.ravel()) * np.sin(t.ravel()),
        center_z + radius * np.cos(p.ravel())))
    return surface, surface  # all visible; mask covers it


def run_case(name, surface, red_band, dim_expect, center_expect, symmetry_expect,
             dim_tolerance=0.005):
    scene = to_camera(np.vstack((table_points(), surface)))
    mask = to_camera(red_band)
    built = build_generic_observation(mask, scene, T_BASE_CAMERA,
                                      stamp={'sec': 1, 'nanosec': 0},
                                      camera_frame='camera_depth_frame')
    if built['status'] != 'ok':
        check(name, False, f'rejected: {built["reason"]}')
        return None
    obs = built['observation']
    dims = np.asarray(obs['dimensions_m'])
    center = np.asarray(obs['obb_center_base_m'])
    dim_error = np.abs(dims - np.asarray(dim_expect))
    center_error = np.abs(center - np.asarray(center_expect))
    check(f'{name}: dims', np.all(dim_error < dim_tolerance),
          f'dims={np.round(dims * 1e3, 1).tolist()} mm error={np.round(dim_error * 1e3, 1).tolist()} mm')
    check(f'{name}: center', np.all(center_error < 0.005),
          f'center={np.round(center, 4).tolist()} error={np.round(center_error * 1e3, 1).tolist()} mm')
    hypothesis = obs['symmetry']['vertical_axis']['class']
    check(f'{name}: symmetry', hypothesis == symmetry_expect,
          f'class={hypothesis} expect={symmetry_expect}')
    check(f'{name}: hull above table',
          built['artifacts']['hull_vertices'][:, 2].min() >= TABLE_Z - 0.005)
    check(f'{name}: confidence', 0.0 < obs['segmentation_confidence'] <= 1.0,
          f"conf={obs['segmentation_confidence']:.2f}")
    return built


def main():
    # 1. The fixed regression cylinder must reconstruct within tolerance.
    cap, band = cylinder_cloud(0.046, 0.08)
    built = run_case('cylinder46x80', np.vstack((cap, band)), band,
                     [0.046, 0.046, 0.08], [0.45, 0.0, TABLE_Z + 0.04], 'likely')
    check('cylinder46x80: no phantom obstacles',
          built is not None and built['observation'].get('obstacles_base') == [],
          str((built or {}).get('observation', {}).get('obstacles_base')))

    # 1b. A second object inside the ROI becomes an obstacle; the target
    # cluster itself must never appear as one.
    cap2, band2 = cylinder_cloud(0.040, 0.05, center_xy=(0.45, 0.055))
    built = run_case('cylinder_with_neighbour', np.vstack((cap, band, cap2, band2)), band,
                     [0.046, 0.046, 0.08], [0.45, 0.0, TABLE_Z + 0.04], 'likely')
    obstacles = (built or {}).get('observation', {}).get('obstacles_base') or []
    check('neighbour detected as one obstacle', len(obstacles) == 1,
          str([[round(v, 3) for v in ob['aabb_min_m']] for ob in obstacles]))
    if obstacles:
        target_inside = all(
            np.asarray(ob['aabb_min_m'])[0] < 0.42 and np.asarray(ob['aabb_max_m'])[0] > 0.44
            for ob in obstacles)
        check('target cluster not listed as obstacle', not target_inside)

    # 2. The fixed regression cube: equal horizontal dims must NOT read as
    # continuous rotational symmetry; the square side band says unlikely.
    top, sides = box_cloud([0.046, 0.046, 0.046])
    run_case('cube46', np.vstack((top, sides)), sides,
             [0.046, 0.046, 0.046], [0.45, 0.0, TABLE_Z + 0.023], 'unlikely')

    # 2b. A yawed square block: PCA dims stay equal at any yaw, so this is
    # the counterexample for dims-only symmetry inference.
    solid = full_box_cloud([0.046, 0.046, 0.05])
    yaw = np.array([[math.cos(0.5), -math.sin(0.5), 0.0],
                    [math.sin(0.5), math.cos(0.5), 0.0],
                    [0.0, 0.0, 1.0]])
    pivot = np.array([0.45, 0.0, TABLE_Z])
    yawed = (solid - pivot) @ yaw.T + pivot
    # PCA-OBB yaw is degenerate for a square, so reported dims may overestimate
    # up to ~sqrt(2)x; the world-frame AABB legitimately grows with yaw.
    # Collision uses the hull itself and revalidation uses local widths, so
    # neither inherits this reporting bias.
    run_case('yawed_square46', yawed, yawed,
             [0.046, 0.046, 0.05], [0.45, 0.0, TABLE_Z + 0.025], 'unlikely',
             dim_tolerance=0.02)

    # 3. A sphere: horizontal dims equal, height from support plane.
    surface, mask_pts = sphere_cloud(0.040)
    run_case('sphere40', surface, mask_pts,
             [0.040, 0.040, 0.040], [0.45, 0.0, TABLE_Z + 0.02], 'likely')

    # 4. A random rectangular block: not axisymmetric, bar-like footprint.
    top, sides = box_cloud([0.060, 0.028, 0.050])
    run_case('box60x28x50', np.vstack((top, sides)), sides,
             [0.060, 0.028, 0.050], [0.45, 0.0, TABLE_Z + 0.025], 'unlikely')

    # 5. Rejections: too few points, and a mask split across three clusters.
    tiny = to_camera(np.vstack((table_points(), cap)))[:120]
    built = build_generic_observation(to_camera(band)[:10], tiny, T_BASE_CAMERA,
                                      table_height_prior_m=TABLE_Z)
    check('reject: insufficient mask points', built['status'] == 'rejected',
          built.get('reason', ''))

    third = len(band) // 3
    blobs_camera = np.vstack((
        to_camera(band[:third]),
        to_camera(band[third:2 * third]) + np.array([0.12, 0.0, 0.0]),
        to_camera(band[2 * third:]) + np.array([-0.12, 0.0, 0.0]),
    ))
    scene = to_camera(np.vstack((table_points(), cap)))
    built = build_generic_observation(blobs_camera, scene, T_BASE_CAMERA,
                                      table_height_prior_m=TABLE_Z)
    check('reject: mask split across clusters', built['status'] == 'rejected',
          built.get('reason', ''))

    # 6. Support-plane prior fallback: a tall object-only cloud has no
    # dominant plane, so the labelled prior must be used and recorded.
    tall = full_box_cloud([0.046, 0.046, 0.20])
    built = build_generic_observation(to_camera(tall), to_camera(tall), T_BASE_CAMERA,
                                      stamp={'sec': 2, 'nanosec': 0},
                                      table_height_prior_m=TABLE_Z,
                                      max_height_m=0.25)
    detail = (built['observation']['quality']['table_source'] if built['status'] == 'ok'
              else built.get('reason', ''))
    check('prior fallback', built['status'] == 'ok' and detail == 'prior_fallback', detail)
    if built['status'] == 'ok':
        height = built['observation']['dimensions_m'][2]
        check('prior fallback height', abs(height - 0.20) < 0.005, f'{height:.3f} m')

    # 7. PLY round trip and saved artifact reload.
    built = build_generic_observation(to_camera(band), to_camera(np.vstack((table_points(), cap, band))),
                                      T_BASE_CAMERA, stamp={'sec': 4, 'nanosec': 500000000})
    check('build for artifact round trip', built['status'] == 'ok', built.get('reason', ''))
    if built['status'] == 'ok':
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_ply(tmp / 'hull.ply', built['artifacts']['hull_vertices'], built['artifacts']['hull_faces'])
            vertices, faces = read_ply(tmp / 'hull.ply')
            check('ply round trip', np.allclose(vertices, built['artifacts']['hull_vertices'], atol=1e-9)
                  and np.array_equal(faces, built['artifacts']['hull_faces']))
            built['artifacts']['mask'] = np.ones((4, 4), dtype=bool)
            observation = save_generic_observation(tmp, 'frame', built)
            points, table = load_object_points(tmp / 'frame_object_points.npz')
            check('artifact reload', len(points) == built['observation']['quality']['points_after_filter']
                  and abs(table - TABLE_Z) < 1e-9)
            check('artifact paths filled', observation['convex_hull_path'] == 'frame_object_hull.ply'
                  and observation['points_base_path'] == 'frame_object_points.npz')
            parsed = GenericObjectObservation.from_dict(observation)
            check('observation parse round trip',
                  np.allclose(parsed.center(), built['observation']['obb_center_base_m'], atol=1e-9))
            check('hold mode from symmetry', parsed.hold_check_mode() == 'upright_tilt')
            (tmp / 'obs.json').write_text(json.dumps(parsed.to_dict()))
            reloaded = GenericObjectObservation.from_dict(
                json.loads((tmp / 'obs.json').read_text()))
            check('json observation round trip', reloaded == parsed)

    # 8. Shared candidate filter behaviour.
    target = np.array([[0.0, 0.0, 0.05], [0.0, 0.01, 0.05], [0.0, -0.01, 0.05]])
    rotation = np.eye(3)
    inside = points_in_jaw_volume(target, np.array([0.0, 0.0, 0.05]), rotation,
                                  width=0.04, height=0.04, depth=0.0)
    check('jaw volume predicate', bool(inside.all()))
    outside = points_in_jaw_volume(target + np.array([0.0, 0.05, 0.0]),
                                   np.array([0.0, 0.0, 0.05]), rotation,
                                   width=0.04, height=0.04, depth=0.0)
    check('jaw volume rejects offset points', not bool(outside.any()))
    payload = {
        'source': 'graspnet_sim_rgbd', 'frame': 'base_link',
        'candidates': [
            {'id': 0, 'opening_m': 0.05,
             'T_base_tcp': np.eye(4).tolist(), 'T_base_pregrasp_tcp': _pregrasp(0.1)},
            {'id': 1, 'opening_m': 0.09,
             'T_base_tcp': np.eye(4).tolist(), 'T_base_pregrasp_tcp': _pregrasp(0.1)},
        ],
    }
    valid = validate_candidates_payload(payload)
    check('payload validation drops wide grasps', [c['id'] for c in valid] == [0])
    check('neighborhood radius', abs(neighborhood_radius_m([0.046, 0.046, 0.08]) -
                                     (0.5 * math.sqrt(0.046 ** 2 + 0.046 ** 2 + 0.08 ** 2) + 0.04)) < 1e-9)
    check('neighborhood accepts near', candidate_center_near_target(
        {'T_base_tcp': _translation(0.45, 0.0, -0.11)}, [0.45, 0.0, -0.11], 0.09))
    check('neighborhood rejects far', not candidate_center_near_target(
        {'T_base_tcp': _translation(0.45, 0.0, 0.1)}, [0.45, 0.0, -0.11], 0.09))

    # 9. Adaptive gripper policy.
    from piperx_lab.gripper_policy import (classify_grasp, closing_sequence,
                                           close_width_for, pre_open_width)
    seq = closing_sequence(0.024)
    check('closing sequence starts outside object', seq[0] >= 0.024 + 0.003, f'{seq}')
    check('closing sequence ends at floor', seq[-1] <= 0.024 - 0.01, f'{seq[-1]:.4f}')
    check('closing sequence monotonically decreasing',
          all(b < a for a, b in zip(seq, seq[1:])))
    check('closing sequence small part bounded', closing_sequence(0.010)[-1] >= 0.008)
    check('closing sequence unknown width', len(closing_sequence(None)) == 2)
    check('close width small object', abs(close_width_for(
        {'predicted_width_m': 0.020}) - 0.018) < 1e-9)
    check('close width large object', abs(close_width_for(
        {'predicted_width_m': 0.060}) - 0.054) < 1e-9)
    check('close width clamped', close_width_for(
        {'predicted_width_m': 0.004}) >= 0.008)
    check('close width fallback uses opening', abs(close_width_for(
        {'opening_m': 0.050}) - 0.044) < 1e-9)
    check('pre open margin', abs(pre_open_width(
        {'opening_m': 0.030, 'predicted_width_m': 0.028}) - 0.038) < 1e-9)
    state, _ = classify_grasp(
        {'closing': {'gripper_link1': True, 'gripper_link2': True},
         'lifting': {'gripper_link1': True, 'gripper_link2': True}},
        [0.01, -0.01], 0.02, lift_m=0.09)
    check('state object_grasped', state == 'object_grasped', state)
    state, _ = classify_grasp({}, None, 0.02, lift_m=0.0)
    check('state empty_grip', state == 'empty_grip', state)
    state, _ = classify_grasp(
        {'closing': {'gripper_link1': True, 'gripper_link2': False}},
        None, 0.02, lift_m=0.05)
    check('state single_sided', state == 'single_sided', state)
    state, _ = classify_grasp(
        {'closing': {'gripper_link1': True, 'gripper_link2': True},
         'lifting': {}},
        None, 0.02, lift_m=0.02)
    check('state slipped', state == 'slipped', state)
    state, _ = classify_grasp({}, [0.001, -0.001], 0.02, lift_m=0.0)
    check('state blocked_unknown', state == 'blocked_unknown', state)

    # 10. Candidate revalidation against a synthetic cloud.
    from piperx_lab.grasp_candidates import apply_revalidation, geometric_candidates
    cap, band = cylinder_cloud(0.046, 0.08)
    cloud = np.vstack((cap, band))
    center = np.array([0.45, 0.0, TABLE_Z + 0.04])
    def tip_pose(approach_deg, grip_z):
        azimuth = math.radians(approach_deg)
        approach = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
        opening = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
        transverse = np.cross(approach, opening)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((transverse, opening, approach))
        pose[:3, 3] = [center[0], center[1], grip_z]
        return pose
    good = {'id': 0, 'source': 'graspnet_rgbd', 'method': 'network', 'score': 1.0,
            'T_base_tip': tip_pose(0, center[2]).tolist(),
            'opening_m': 0.052, 'approach_distance_m': 0.08}
    tip = np.asarray(good['T_base_tip'])
    pre = tip.copy(); pre[:3, 3] -= 0.08 * tip[:3, 2]
    good['T_base_tcp'] = good['T_base_pregrasp_tcp'] = pre.tolist()
    validated = apply_revalidation([good], cloud)
    check('revalidate accepts side grasp', validated[0]['rejection_reason'] is None,
          validated[0]['rejection_reason'])
    check('revalidate predicts width', abs(validated[0]['predicted_width_m'] - 0.046) < 0.004,
          f"{validated[0].get('predicted_width_m')}")
    too_wide = dict(good, id=1, opening_m=0.069,
                    T_base_tip=tip_pose(90, center[2]).tolist())
    tip2 = np.asarray(too_wide['T_base_tip'])
    pre2 = tip2.copy(); pre2[:3, 3] -= 0.08 * tip2[:3, 2]
    too_wide['T_base_tcp'] = too_wide['T_base_pregrasp_tcp'] = pre2.tolist()
    # An 80 mm wide block cannot fit the 70 mm opening across its long axis.
    wide_top, wide_sides = box_cloud([0.080, 0.030, 0.050])
    wide_cloud = np.vstack((wide_top, wide_sides))
    validated = apply_revalidation([too_wide], wide_cloud)
    check('revalidate rejects infeasible width',
          validated[0]['rejection_reason'] == 'width_infeasible_for_piper_opening',
          validated[0]['rejection_reason'])
    off = dict(good, id=2, T_base_tip=_translation(0.45, 0.0, TABLE_Z + 0.16))
    validated = apply_revalidation([off], cloud)
    check('revalidate rejects off-target tip',
          validated[0]['rejection_reason'] is not None,
          validated[0]['rejection_reason'])

    # 11. Geometric candidates from the observation only (no colour).
    observation = {
        'dimensions_m': [0.046, 0.046, 0.08],
        'obb_center_base_m': [0.45, 0.0, TABLE_Z + 0.04],
        'stamp': {'sec': 1, 'nanosec': 0},
    }
    generated = geometric_candidates(observation, cloud, table_height_m=TABLE_Z, limit=8)
    check('geometric candidates generated', len(generated) >= 2, f'n={len(generated)}')
    check('geometric candidates carry source',
          all(c['source'] == 'geometric_antipodal' and c['method'] == 'geometry'
              for c in generated))
    check('geometric candidates carry stamp',
          all(c['stamp'] == {'sec': 1, 'nanosec': 0} for c in generated))
    check('geometric candidates validated',
          all(c.get('rejection_reason') is None and c.get('confidence') is not None
              for c in generated))

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: {FAILURES}')
        return 1
    print('All synthetic observation tests passed.')
    return 0


def _pregrasp(distance):
    matrix = np.eye(4)
    matrix[0, 3] = distance
    return matrix.tolist()


def _translation(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix.tolist()


if __name__ == '__main__':
    raise SystemExit(main())
