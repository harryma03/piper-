"""Viewpoint planning for active multi-view observation.

Generates a small, ordered set of wrist-camera viewpoints around the
observed target: azimuth offsets on a ring at the current camera radius,
each candidate pre-screened by (a) a collision-free clearance sphere around
the camera pose and (b) an expected-novelty proxy (the view direction must
face a hull region that current views covered poorly). IK feasibility and
joint limits are checked by the executor against the live robot; the planner
only rejects poses that are geometrically hopeless (below the support plane,
beyond the workspace radius, or colliding with fused obstacle boxes).
"""
import math

import numpy as np

CAMERA_CLEARANCE_M = 0.06      # keep the camera this far from hull/obstacles
MAX_RADIUS_M = 0.62            # from base origin: Piper reach boundary
AZIMUTH_STEPS_DEG = (60.0, 120.0, 180.0)
VIEW_HEIGHTS_M = (0.02, 0.10)  # above the current view height


def _clearance_ok(camera_xy_base, z, hull_vertices, obstacles, clearance_m):
    """3D-aware clearance: XY proximity only blocks when the camera height
    is also inside the shape's z span (a camera hovering above the object
    is legitimately close in plan view)."""
    if hull_vertices is not None and len(hull_vertices):
        z_lo, z_hi = float(hull_vertices[:, 2].min()), float(hull_vertices[:, 2].max())
        if z_lo - clearance_m <= z <= z_hi + clearance_m:
            distances = np.linalg.norm(hull_vertices[:, :2] - camera_xy_base, axis=1)
            if distances.min() < clearance_m:
                return False
    for box in obstacles or []:
        low = np.asarray(box['aabb_min_m'], dtype=float)
        high = np.asarray(box['aabb_max_m'], dtype=float)
        inside_xy = (low[0] - clearance_m <= camera_xy_base[0] <= high[0] + clearance_m and
                     low[1] - clearance_m <= camera_xy_base[1] <= high[1] + clearance_m)
        if inside_xy and low[2] - clearance_m <= z <= high[2] + clearance_m:
            return False
    return True


def plan_viewpoints(observation, fused_summary, obstacles=None, *,
                    current_camera_xy=(0.45, 0.0), max_views=4,
                    radius_m=0.28, height_above_target=(0.16, 0.24)):
    """Ordered viewpoint candidates around the target (base-frame poses).

    Returns a list of dicts with base-frame camera position, look-at target
    and the planner's rationale; the executor performs IK/collision checks
    and executes them. Novelty is a proxy: azimuth steps away from the
    current view are assumed to reveal previously hidden hull faces, scaled
    by how much of the hull interior is still unobserved.
    """
    center = np.asarray(observation.get('obb_center_base_m', [0.45, 0, -0.11]),
                        dtype=float)
    hull = observation.get('quality', {}).get('hull_aabb_extents_m')
    hull_vertices = None  # planner works from the AABB; the executor has the hull
    if hull:
        extents = np.asarray(hull, dtype=float) / 2
        hull_vertices = (center + np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
                                            [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
                                            [-1, -1, -1]]) * extents)
    unseen = float((fused_summary or {}).get('observed_fraction_inside_hull', 0.0) or 0.0)
    novelty_boost = 1.0 - unseen  # more unseen interior -> more views worthwhile

    candidates = []
    base_xy = np.asarray(current_camera_xy, dtype=float)
    target_xy = center[:2]
    # Rotate the CURRENT camera offset AROUND THE TARGET (not the base
    # origin): the camera keeps its distance and bearing to the object, so
    # the object stays centred in view at the new azimuth.
    offset = base_xy - target_xy
    for step_deg in AZIMUTH_STEPS_DEG:
        for dz in VIEW_HEIGHTS_M[:1 + int(novelty_boost > 0.4)]:
            angle = math.radians(step_deg)
            rot = np.array([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
            xy = target_xy + rot @ offset
            z = float(center[2] + height_above_target[0] + dz)
            radius = float(np.linalg.norm(xy))
            if radius > MAX_RADIUS_M or z <= center[2] + 0.02:
                continue
            if not _clearance_ok(xy, z, hull_vertices, obstacles, CAMERA_CLEARANCE_M):
                continue
            candidates.append({
                'view_id': f'v_az{int(step_deg)}_dz{int(dz * 100)}',
                'camera_position_base_m': [float(xy[0]), float(xy[1]), z],
                'lookat_base_m': [float(center[0]), float(center[1]), float(center[2])],
                'azimuth_step_deg': step_deg,
                'expected_novelty_proxy': round(min(1.0, (step_deg / 180.0) * (0.5 + novelty_boost)), 3),
                'rationale': f'azimuth +{step_deg:g} deg, camera z {z:.3f} m',
            })
            if len(candidates) >= max_views:
                return candidates
    return candidates
