"""Shared GraspNet-to-Piper candidate filtering.

Candidate-level checks used by both candidate preparation and execution live
here so per-object callers cannot drift apart: Piper opening limits, jaw
target association, and a target-neighbourhood sanity check. MoveIt-stage
validation (IK, paths, staged contact) stays with the executor, which is
already object-agnostic.
"""
import math

import numpy as np

# Target widths, not measured contact forces; the simulated Piper opens 70 mm.
MAX_OPENING_M = 0.070
OPENING_MARGIN_M = 0.004
# GraspNet's idealized finger extends from x=depth-0.06 to x=depth.
GRASPNET_FINGER_LENGTH_M = 0.06
APPROACH_DISTANCES_M = (0.10, 0.08, 0.06)
MAX_CANDIDATES = 40


def points_in_jaw_volume(points, center, rotation, width, height, depth,
                         tolerance_m=0.0, finger_length_m=GRASPNET_FINGER_LENGTH_M):
    """Boolean mask of points inside the predicted jaw volume.

    GraspNet frame: +X approach, +Y opening, +Z transverse; the jaw runs from
    ``depth - finger_length_m`` to ``depth`` along local X. Deliberately a
    coarse target filter, not a contact proof.
    """
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or rotation.shape != (3, 3):
        raise ValueError('jaw volume needs Nx3 points and a 3x3 rotation')
    local = (points - center) @ rotation
    return ((local[:, 0] >= depth - finger_length_m - tolerance_m) &
            (local[:, 0] <= depth + tolerance_m) &
            (np.abs(local[:, 1]) <= width / 2 + tolerance_m) &
            (np.abs(local[:, 2]) <= height / 2 + tolerance_m))


def validate_candidates_payload(data, *, max_opening_m=MAX_OPENING_M):
    """Validate a merged GraspNet candidate payload and drop over-wide grasps.

    Moved from the simulation runner so preparation and execution enforce the
    identical structure and opening rules. Returns the width-compatible
    candidates; raises ValueError on structural problems.
    """
    if data.get('source') not in ('graspnet_sim_rgbd', 'graspnet_fusion') or data.get('frame') != 'base_link':
        raise ValueError('Expected simulated RGB-D GraspNet candidates in base_link')
    candidates = data.get('candidates')
    if not isinstance(candidates, list):
        raise ValueError('Candidate payload lacks a candidates list')
    if not candidates:
        # Structured reason: every candidate was already rejected upstream
        # (revalidation/rotation), so the executor can attribute the failure
        # precisely instead of a generic count error.
        raise ValueError('empty_candidate_payload: no candidate survived upstream revalidation')
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f'Expected 1 to {MAX_CANDIDATES} GraspNet candidates')
    valid, seen_ids = [], set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError('Each GraspNet candidate must be an object')
        ident = candidate.get('id')
        if not isinstance(ident, int) or isinstance(ident, bool) or ident in seen_ids:
            raise ValueError('Candidate IDs must be unique integers')
        seen_ids.add(ident)
        width = float(candidate.get('opening_m', float('nan')))
        if not math.isfinite(width) or not 0 < width <= max_opening_m:
            # Model grippers can be wider than the simulated Piper gripper.
            continue
        matrices = []
        for key in ('T_base_tcp', 'T_base_pregrasp_tcp'):
            matrix = np.asarray(candidate.get(key), dtype=float)
            if (matrix.shape != (4, 4) or not np.all(np.isfinite(matrix))
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-4)):
                raise ValueError(f'Candidate {ident}: invalid {key}')
            matrices.append(matrix)
        distance = np.linalg.norm(matrices[0][:3, 3] - matrices[1][:3, 3])
        if (not 0.01 <= distance <= 0.2
                or not np.allclose(matrices[0][:3, :3], matrices[1][:3, :3], atol=1e-4)):
            raise ValueError(f'Candidate {ident}: invalid pregrasp offset or orientation')
        valid.append(candidate)
    if not valid:
        raise ValueError('No candidates fit the simulated gripper opening')
    return valid


def neighborhood_radius_m(dimensions_m, margin_m=0.04):
    """Target-neighbourhood radius from observed dimensions: half diagonal + margin."""
    dimensions = np.asarray(dimensions_m, dtype=float)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError('neighborhood radius needs positive observed dimensions')
    return float(0.5 * np.linalg.norm(dimensions) + margin_m)


def candidate_center_near_target(candidate, target_center_base, radius_m):
    """Whether the predicted grip centre sits in a reasonable target neighbourhood.

    The grip centre is the fingertip midpoint (``T_base_tip``); the flange
    ``T_base_tcp`` trails it by the model tip offset and is not near the
    object by design. A coarse sanity filter on top of jaw association; it
    does not validate contact geometry.
    """
    frame = candidate.get('T_base_tip', candidate.get('T_base_tcp'))
    pose = np.asarray(frame, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError('candidate lacks a valid T_base_tip or T_base_tcp')
    center = np.asarray(target_center_base, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('target center must be 3 finite numbers')
    return float(np.linalg.norm(pose[:3, 3] - center)) <= float(radius_m)
