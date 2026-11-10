"""MoveIt collision geometry from a generic observation's convex hull.

The target collision body comes from the observed convex hull, never from
hard-coded sizes. Validation rejects low-quality geometry instead of falling
back to assumed dimensions. The support plane is added separately by the
caller so the table is never merged into the target body.
"""
import math

import numpy as np

from piperx_lab.object_observation import ObservationError, read_ply

MIN_HULL_DIMENSION_M = 0.015
MAX_HULL_DIMENSION_M = 0.20
MIN_HULL_VERTICES = 4
MAX_HULL_VERTICES = 4096
TABLE_PENETRATION_TOLERANCE_M = 0.005
DIMENSION_AGREEMENT_TOLERANCE_M = 0.005


def load_validated_hull(hull_path, *, table_height_m, observation=None):
    """Load a hull PLY and enforce geometry quality gates.

    Returns ``(vertices, faces, provenance)``. Raises ObservationError on
    any gate failure; callers must refuse to grasp rather than substitute
    assumed sizes.
    """
    vertices, faces = read_ply(hull_path)
    if len(vertices) < MIN_HULL_VERTICES or len(faces) < MIN_HULL_VERTICES:
        raise ObservationError(
            f'Hull {hull_path} has {len(vertices)} vertices/{len(faces)} faces')
    if len(vertices) > MAX_HULL_VERTICES:
        raise ObservationError(f'Hull {hull_path} exceeds {MAX_HULL_VERTICES} vertices')
    if not np.all(np.isfinite(vertices)):
        raise ObservationError(f'Hull {hull_path} contains non-finite vertices')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ObservationError(f'Hull {hull_path} has out-of-range face indices')

    extents = np.ptp(vertices, axis=0)
    if (np.any(extents < MIN_HULL_DIMENSION_M) or np.any(extents > MAX_HULL_DIMENSION_M)):
        raise ObservationError(
            f'Hull extents {extents.tolist()} outside '
            f'[{MIN_HULL_DIMENSION_M}, {MAX_HULL_DIMENSION_M}] m')
    if vertices[:, 2].min() < table_height_m - TABLE_PENETRATION_TOLERANCE_M:
        raise ObservationError(
            f'Hull bottom {vertices[:, 2].min():.4f} m is below the support plane '
            f'{table_height_m:.4f} m by more than {TABLE_PENETRATION_TOLERANCE_M * 1e3:.0f} mm')
    if observation is not None:
        quality = observation.get('quality') or {}
        aabb = quality.get('hull_aabb_extents_m') or quality.get('aabb_extents_m')
        if aabb:
            # Both are AABBs of the same filtered cloud; a mismatch means the
            # hull file does not belong to this observation.
            delta = np.abs(extents - np.asarray(aabb, dtype=float))
            if np.any(delta > 1e-6):
                raise ObservationError(
                    f'Hull extents {extents.tolist()} disagree with the observation '
                    f'AABB {list(aabb)}; the hull file may not match this observation')
        dims = np.asarray(observation.get('dimensions_m'), dtype=float)
        if dims.shape == (3,) and extents[2] > dims[2] + DIMENSION_AGREEMENT_TOLERANCE_M:
            # The observed supported height includes the hidden bottom, so it
            # bounds the visible hull height from above.
            raise ObservationError(
                f'Hull height {extents[2]:.4f} m exceeds the observed supported '
                f'height {dims[2]:.4f} m')

    edges = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edges[key] = edges.get(key, 0) + 1
    nonmanifold = sum(1 for count in edges.values() if count != 2)
    if nonmanifold:
        raise ObservationError(
            f'Hull {hull_path} is not closed: {nonmanifold} edges are not shared by two faces')

    provenance = {
        'source': 'observation_convex_hull',
        'path': str(hull_path),
        'vertex_count': int(len(vertices)),
        'triangle_count': int(len(faces)),
        'closed': True,
        'extents_m': [float(v) for v in extents],
        'statistically_filtered': True,
        'observation_stamp': dict(observation.get('stamp') or {}) if observation else None,
    }
    return vertices, faces, provenance


def collision_object_from_hull(hull_path, *, table_height_m, observation=None,
                               object_id='target', frame_id='base_link'):
    """Build a MoveIt mesh CollisionObject from the validated hull.

    Returns ``(CollisionObject, provenance)``. ROS shape_msgs are imported
    lazily so the validation half stays testable without a ROS environment.
    """
    from shape_msgs.msg import Mesh, MeshTriangle  # noqa: PLC0415 - requires ROS env
    from geometry_msgs.msg import Point  # noqa: PLC0415

    vertices, faces, provenance = load_validated_hull(
        hull_path, table_height_m=table_height_m, observation=observation)
    mesh = Mesh()
    mesh.vertices = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in vertices]
    mesh.triangles = [MeshTriangle(vertex_indices=[int(a), int(b), int(c)]) for a, b, c in faces]
    from moveit_msgs.msg import CollisionObject  # noqa: PLC0415
    obj = CollisionObject(id=object_id, operation=CollisionObject.ADD)
    obj.header.frame_id = frame_id
    obj.meshes = [mesh]
    from geometry_msgs.msg import Pose  # noqa: PLC0415
    obj.mesh_poses = [Pose()]
    return obj, provenance
