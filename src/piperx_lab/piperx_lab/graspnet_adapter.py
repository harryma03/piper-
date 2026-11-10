"""Convert observed GraspNet candidates to simulated Piper TCP targets.

No ROS calls or actuator commands are made here.  GraspNet's default gripper
model and the Piper fingers are geometrically different: this provides an
initial alignment for subsequent robot-specific collision and path checks,
not a guarantee of physical contact or a stable grasp.
"""

import math
import numpy as np

from piperx_lab.grasp_candidate_filter import (
    OPENING_MARGIN_M,
    APPROACH_DISTANCES_M,
    MAX_OPENING_M,
    points_in_jaw_volume,
)


def _transform(value, label):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f'{label} must be a finite rigid 4x4 transform')
    return matrix


def convert_grasps(predictions, T_base_camera, T_link6_tip, T_link6_tcp,
                   target_points_camera,
                   approach_distances=APPROACH_DISTANCES_M,
                   max_opening=MAX_OPENING_M, opening_margin=OPENING_MARGIN_M,
                   min_target_points=3, target_tolerance=0.005):
    """Convert Nx17 decoded grasps and reject those lacking target support.

    GraspNet axes: +X approach, +Y opening, +Z transverse. Its idealized
    finger extends from x=depth-0.06 to x=depth (collision_detector.py).
    Piper's fingertip midpoint is aligned to that x=depth point, and its +Z
    faces the approach direction, +Y is the opening axis. Red surface points
    inside the predicted jaw volume associate the grasp with the observed
    target; that test is deliberately only a coarse target filter.

    Candidates keep the sampled image's timestamp and arm state externally.
    `opening_m` is the commanded *total* width; max_opening=0.07 is the
    current simulation runner's tested opening, not a physical limit claim.
    """
    rows = np.asarray(predictions, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 17:
        raise ValueError('predictions must be an Nx17 GraspNet array')
    T_base_camera = _transform(T_base_camera, 'T_base_camera')
    T_link6_tip = _transform(T_link6_tip, 'T_link6_tip')
    T_link6_tcp = _transform(T_link6_tcp, 'T_link6_tcp')
    target = np.asarray(target_points_camera, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError('target_points_camera must have shape Nx3')
    target = target[np.isfinite(target).all(axis=1)]
    if len(target) < min_target_points or min_target_points < 1:
        raise ValueError('insufficient finite target surface points')
    if (not np.isfinite([max_opening, opening_margin, target_tolerance]).all() or
            max_opening <= 0 or opening_margin < 0 or target_tolerance < 0):
        raise ValueError('invalid opening or tolerance')
    distances = [float(d) for d in approach_distances]
    if not distances or any(not math.isfinite(d) or d <= 0 for d in distances):
        raise ValueError('approach distances must be positive and finite')

    # Columns in GraspNet coordinates are Piper tip frame X, Y, Z.
    R_grasp_tip = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    T_tip_tcp = np.linalg.inv(T_link6_tip) @ T_link6_tcp
    accepted, rejected = [], []
    for model_id, row in enumerate(rows):
        score, width, height, depth = row[:4]
        rotation = row[4:13].reshape(3, 3)
        center = row[13:16]
        reason = None
        if (not np.isfinite(row).all() or width <= 0 or height <= 0 or
                depth <= 0 or depth > 0.06 or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            reason = 'invalid_grasp_geometry'
        elif width + opening_margin > max_opening + 1e-9:
            reason = 'opening_exceeds_simulation_limit'
        else:
            inside = points_in_jaw_volume(target, center, rotation, width, height,
                                          depth, tolerance_m=target_tolerance)
            count = int(inside.sum())
            if count < min_target_points:
                reason = 'insufficient_observed_target_in_jaw_volume'
        if reason:
            rejected.append({'model_id': model_id, 'reason': reason})
            continue
        T_camera_tip = np.eye(4)
        T_camera_tip[:3, :3] = rotation @ R_grasp_tip
        T_camera_tip[:3, 3] = center + rotation[:, 0] * depth
        T_base_tip = T_base_camera @ T_camera_tip
        tcp = T_base_tip @ T_tip_tcp
        approach = T_base_tip[:3, 2]
        for distance in distances:
            pre_tip = T_base_tip.copy()
            pre_tip[:3, 3] -= distance * approach
            pre_tcp = pre_tip @ T_tip_tcp
            accepted.append({'id': len(accepted), 'model_id': model_id,
                             'kind': 'graspnet_rgbd', 'score': float(score),
                             'predicted_width_m': float(width),
                             'opening_m': float(width + opening_margin),
                             'predicted_depth_m': float(depth),
                             'approach_distance_m': distance,
                             'target_points_in_jaw_volume': count,
                             'target_jaw_tolerance_m': target_tolerance,
                             'T_base_tip': T_base_tip.tolist(),
                             'T_base_tcp': tcp.tolist(),
                             'T_base_pregrasp_tcp': pre_tcp.tolist(),
                             'approach_direction': approach.tolist(),
                             'graspnet_tip_alignment': 'finger_end_at_depth',
                             'contact_point_independently_validated': False})
    return accepted, rejected
