#!/usr/bin/env python3
"""Foundation tests for the closed-loop grasp refactor.

Covers the audit findings: right-handed geometric-candidate rotations,
the fingertip-to-TCP model transform, round-trip transform consistency,
and the structured failure reason for empty candidate payloads.
ROS-dependent behaviour (attached-payload planning states) is asserted in
the executor report instead.
"""
from pathlib import Path
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
from piperx_lab.grasp_candidates import (  # noqa: E402
    _approach_rotation,
    geometric_candidates,
    revalidate_candidate,
)
from piperx_lab.grasp_candidate_filter import validate_candidates_payload  # noqa: E402
from piperx_lab.gripper_policy import classify_grasp  # noqa: E402

FAILURES = []


def check(name, condition, detail=''):
    print(f"[{'ok' if condition else 'FAIL'}] {name} {detail}", flush=True)
    if not condition:
        FAILURES.append(name)


def main():
    # 1. Right-handed rotations for every azimuth/tilt combination.
    bad = []
    for azimuth_deg in range(0, 360, 15):
        for tilt_deg in (0.0, 30.0, 55.0):
            az, tilt = math.radians(azimuth_deg), math.radians(tilt_deg)
            horizontal = np.array([math.cos(az), math.sin(az), 0.0])
            approach = (math.cos(tilt) * horizontal
                        + math.sin(tilt) * np.array([0.0, 0.0, -1.0]))
            opening = np.cross(np.array([0.0, 0.0, 1.0]), horizontal)
            R = _approach_rotation(approach, opening)
            if R is None or not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
                bad.append((azimuth_deg, tilt_deg))
            elif R is not None:
                # Axis semantics: columns are X/Y/Z; Z must stay the approach.
                if not np.allclose(R[:, 2], approach, atol=1e-9):
                    bad.append((azimuth_deg, tilt_deg, 'approach axis'))
                if not np.allclose(R[:, 0], np.cross(R[:, 1], R[:, 2]), atol=1e-9):
                    bad.append((azimuth_deg, tilt_deg, 'not right-handed'))
    check('rotation matrices right-handed with Z=approach', not bad, str(bad[:3]))

    # 2. Round trip: TCP = tip @ T_tip_tcp recovers the tip through the inverse.
    rng = np.random.default_rng(7)
    tip = np.eye(4)
    tip[:3, :3] = _approach_rotation(np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    tip[:3, 3] = [0.45, 0.0, -0.11]
    offset = np.eye(4)
    offset[:3, 3] = [0.0, 0.0, -0.1425]  # flange trails along approach (local -Z)
    tcp = tip @ offset
    recovered_tip = tcp @ np.linalg.inv(offset)
    check('tip/tcp round trip', np.allclose(recovered_tip, tip, atol=1e-9))
    check('tcp differs from tip by the model offset',
          abs(np.linalg.norm(tcp[:3, 3] - tip[:3, 3]) - 0.1425) < 1e-6)

    # 3. Geometric candidates apply the TCP transform when given.
    table_z = -0.15
    thetas = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rr = np.linspace(0, 0.023, 15)
    t, r = np.meshgrid(thetas, rr)
    cap = np.column_stack((0.45 + r.ravel() * np.cos(t.ravel()),
                           0.0 + r.ravel() * np.sin(t.ravel()),
                           np.full(t.size, table_z + 0.078)))
    zs = np.linspace(table_z + 0.03, table_z + 0.076, 20)
    t2, z2 = np.meshgrid(thetas, zs)
    side = np.column_stack((0.45 + 0.023 * np.cos(t2.ravel()),
                            0.023 * np.sin(t2.ravel()),
                            z2.ravel()))
    cloud = np.vstack((cap, side))
    observation = {'dimensions_m': [0.046, 0.046, 0.08],
                   'obb_center_base_m': [0.45, 0.0, table_z + 0.04],
                   'stamp': {'sec': 1, 'nanosec': 0}}
    with_transform = geometric_candidates(observation, cloud, table_height_m=table_z,
                                          limit=6, T_tip_tcp=offset)
    without_transform = geometric_candidates(observation, cloud, table_height_m=table_z,
                                             limit=6)
    check('geometric candidates generated with transform', len(with_transform) >= 2,
          f'n={len(with_transform)}')
    if with_transform:
        c = with_transform[0]
        tip_back = np.asarray(c['T_base_tcp']) @ np.linalg.inv(offset)
        check('candidate tcp composes the tip transform',
              np.allclose(tip_back[:3, 3], np.asarray(c['T_base_tip'])[:3, 3], atol=1e-9))
        check('candidate tcp offset is the model tip distance',
              abs(np.linalg.norm(np.asarray(c['T_base_tcp'])[:3, 3]
                                 - np.asarray(c['T_base_tip'])[:3, 3]) - 0.1425) < 1e-6)
        check('tcp_transform_applied flag', c.get('tcp_transform_applied') is True,
              f"flag={c.get('tcp_transform_applied')} "
              f"tip_tcp_delta={np.linalg.norm(np.asarray(c['T_base_tcp'])[:3, 3] - np.asarray(c['T_base_tip'])[:3, 3]):.4f}")
        check('pregrasp trails tcp along approach',
              abs(np.linalg.norm(np.asarray(c['T_base_pregrasp_tcp'])[:3, 3]
                                 - np.asarray(c['T_base_tcp'])[:3, 3])
                  - c['approach_distance_m']) < 1e-6)
    if without_transform:
        check('no-transform candidates flagged',
              all(not c.get('tcp_transform_applied') for c in without_transform))

    # 4. Empty candidate payloads raise a structured, greppable reason.
    try:
        validate_candidates_payload({'source': 'graspnet_sim_rgbd', 'frame': 'base_link',
                                     'candidates': []})
        check('empty payload rejected', False)
    except ValueError as exc:
        check('empty payload structured reason', 'empty_candidate_payload' in str(exc), str(exc))

    # 5. Grasp-state classification distinguishes contact without lift.
    contacts = {'closing': {'gripper_link1': True, 'gripper_link2': True},
                'lifting': {'gripper_link1': True, 'gripper_link2': True}}
    state, _ = classify_grasp(contacts, [0.012, -0.012], 0.03, lift_m=0.00002)
    check('contact_without_lift state', state == 'contact_without_lift', state)
    state, _ = classify_grasp(contacts, [0.012, -0.012], 0.03, lift_m=0.09)
    check('object_grasped on real lift', state == 'object_grasped', state)

    # 6. Multi-view fusion: new viewpoint adds novel voxels; an unrelated
    # cloud is rejected; unseen-interior is tracked.
    from piperx_lab.observation_fusion import FusedObservation, fuse_view, summarize
    fused = FusedObservation()
    radius, zs = 0.023, np.repeat(np.linspace(-0.13, -0.09, 8), 80)
    theta_a = np.tile(np.linspace(0, math.pi, 80), 8)
    theta_b = np.tile(np.linspace(math.pi, 2 * math.pi, 80), 8)
    view_a = np.column_stack((0.45 + radius * np.cos(theta_a),
                              radius * np.sin(theta_a), zs))
    view_b = np.column_stack((0.45 + radius * np.cos(theta_b),
                              radius * np.sin(theta_b), zs))
    merge_a = fuse_view(fused, view_a, 'v0', {'sec': 1, 'nanosec': 0})
    merge_b = fuse_view(fused, view_b, 'v1', {'sec': 2, 'nanosec': 0})
    merge_c = fuse_view(fused, view_b + np.array([0.3, 0.3, 0.3]), 'v2',
                        {'sec': 3, 'nanosec': 0})
    check('first view accepted', merge_a['accepted'])
    check('opposite view contributes novel voxels',
          merge_b['accepted'] and merge_b['novel_voxels'] > 100,
          f"novel={merge_b.get('novel_voxels')}")
    check('unrelated cloud rejected', not merge_c['accepted'],
          merge_c.get('reason'))
    summary = summarize(fused)
    check('fusion tracks views and coverage',
          summary['total_views'] == 2 and summary['observed_voxels'] > 300
          and 0 < summary['observed_fraction_inside_hull'] < 1,
          f"fraction={summary['observed_fraction_inside_hull']}")

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: {FAILURES}')
        return 1
    print('All foundation tests passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
