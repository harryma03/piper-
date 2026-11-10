"""Multi-view observation fusion in the base frame.

Merges per-viewpoint object clouds into one voxel-deduplicated point set,
tracks which voxel was seen from which view/stamp, rebuilds the convex hull
over the union, and labels space as OBSERVED (voxel has returns),
UNSEEN_INSIDE (inside the hull but never observed -- a hole that the support
shadow completion may be hiding) or OUTSIDE. No truth input: every cloud
carries the stamp and camera pose of the view that produced it.

Registration is identity in the base frame (each view is already transformed
by its image-time TF before it arrives here); a sanity check rejects views
whose cloud does not overlap the fused set at all, which catches a stale or
wrong-frame contribution.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

VOXEL_M = 0.0025          # 2.5 mm fusion grid
MIN_NEW_POINTS = 15       # a view must contribute this many novel voxels
OVERLAP_REJECT = 0.02     # <2% overlap with fused set => suspect frame


@dataclass
class FusedObservation:
    """Accumulated multi-view object model (all in base frame)."""
    points: np.ndarray = None              # Nx3 fused, voxel-deduplicated
    voxel_views: dict = field(default_factory=dict)  # voxel key -> [(view_id, stamp)]
    views: list = field(default_factory=list)   # per-view metadata dicts
    hull_vertices: np.ndarray = None
    hull_faces: np.ndarray = None
    hull_volume_m3: float = 0.0
    observed_voxels: int = 0
    hull_inner_unseen_voxels: int = 0      # holes inside the hull
    total_views: int = 0
    rejected_views: list = field(default_factory=list)

    def observed_fraction_inside_hull(self):
        total = self.observed_voxels + self.hull_inner_unseen_voxels
        return self.observed_voxels / total if total else 0.0


def _voxel_keys(points, voxel_m=VOXEL_M):
    return {tuple((p / voxel_m).round().astype(np.int64)) for p in points}


def fuse_view(fused: FusedObservation, points_base, view_id, stamp,
              camera_pose_base=None, min_points=50) -> dict:
    """Merge one view's object cloud; returns merge diagnostics.

    Views that fail the overlap sanity check are recorded and dropped
    instead of silently corrupting the fused model.
    """
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < min_points:
        return {'accepted': False, 'reason': 'insufficient_points',
                'points': int(len(points))}
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < min_points:
        return {'accepted': False, 'reason': 'insufficient_finite_points',
                'points': int(len(points))}

    new_keys = _voxel_keys(points)
    if fused.points is not None and len(fused.points):
        existing = _voxel_keys(fused.points)
        novel = new_keys - existing
        overlap = 1.0 - len(novel) / max(len(new_keys), 1)
        if overlap < OVERLAP_REJECT:
            fused.rejected_views.append({
                'view_id': view_id, 'stamp': dict(stamp or {}),
                'reason': 'no_overlap_with_fused_model',
                'novel_voxels': len(novel), 'overlap': float(overlap)})
            return {'accepted': False, 'reason': 'no_overlap_with_fused_model',
                    'overlap': float(overlap)}
    else:
        novel = new_keys
        overlap = 0.0

    kept = []
    for key in novel:
        kept.append(np.array(key, dtype=float) * VOXEL_M)
    merged_points = kept
    if fused.points is not None and len(fused.points):
        merged_points.insert(0, fused.points)
    fused.points = np.vstack(merged_points)
    fused.observed_voxels = len(fused.points)
    for key in novel:
        fused.voxel_views.setdefault(key, []).append((view_id, dict(stamp or {})))
    fused.total_views += 1

    hull_ok = False
    try:
        hull = ConvexHull(fused.points)
        indices = np.asarray(hull.vertices)
        index_map = {int(old): new for new, old in enumerate(indices)}
        fused.hull_vertices = fused.points[indices]
        fused.hull_faces = np.array([[index_map[int(a)], index_map[int(b)], index_map[int(c)]]
                                     for a, b, c in hull.simplices], dtype=np.int64)
        fused.hull_volume_m3 = float(hull.volume)
        hull_ok = True
    except QhullError:
        fused.hull_vertices = None
        fused.hull_faces = None
        fused.hull_volume_m3 = 0.0

    # Unseen interior: sample the hull bounding box on the fusion grid and
    # count voxels inside the hull with no observation. Coarse by design;
    # it bounds how much the support-shadow completion is guessing.
    fused.hull_inner_unseen_voxels = 0
    if hull_ok and len(fused.points) >= 10:
        low = fused.hull_vertices.min(axis=0)
        high = fused.hull_vertices.max(axis=0)
        steps = np.maximum(((high - low) / VOXEL_M).round().astype(int), 1)
        xs = np.linspace(low[0], high[0], min(steps[0], 40))
        ys = np.linspace(low[1], high[1], min(steps[1], 40))
        zs = np.linspace(low[2], high[2], min(steps[2], 40))
        grid = np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3)
        # A point is inside the hull iff it satisfies every half-space
        # equation (ConvexHull has no find_simplex; equations do).
        equations = hull.equations  # Mx4 rows: [nx, ny, nz, offset] <= 0 inside
        inside = np.all(grid @ equations[:, :3].T + equations[:, 3] <= 1e-9, axis=1)
        observed_keys = _voxel_keys(fused.points)
        unseen = 0
        for p in grid[inside]:
            key = tuple((p / VOXEL_M).round().astype(np.int64))
            if key not in observed_keys:
                unseen += 1
        fused.hull_inner_unseen_voxels = unseen

    fused.views.append({
        'view_id': view_id, 'stamp': dict(stamp or {}),
        'camera_pose_base': None if camera_pose_base is None
        else np.asarray(camera_pose_base, dtype=float).tolist(),
        'points_raw': int(len(points)),
        'novel_voxels': int(len(novel)),
        'overlap_with_fused': float(overlap),
    })
    return {'accepted': True, 'novel_voxels': int(len(novel)),
            'overlap': float(overlap), 'hull_ok': hull_ok,
            'observed_voxels': fused.observed_voxels,
            'unseen_inside_hull': fused.hull_inner_unseen_voxels}


def summarize(fused: FusedObservation) -> dict:
    """JSON-safe summary for reports and the perception service."""
    return {
        'total_views': fused.total_views,
        'observed_voxels': fused.observed_voxels,
        'hull_volume_m3': fused.hull_volume_m3,
        'unseen_inside_hull_voxels': fused.hull_inner_unseen_voxels,
        'observed_fraction_inside_hull': round(fused.observed_fraction_inside_hull(), 4),
        'rejected_views': fused.rejected_views,
        'views': [{k: v for k, v in view.items() if k != 'camera_pose_base'}
                  for view in fused.views],
        'points': None if fused.points is None else len(fused.points),
    }
