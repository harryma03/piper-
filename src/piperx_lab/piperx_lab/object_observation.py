"""Generic shape-agnostic RGB-D object observation.

Turns a color-mask point cloud plus the full-scene depth cloud into a shared
target description: support-plane estimate, euclidean cluster, statistical
filtering, convex hull, oriented box, dimensions and a symmetry hypothesis.

No fixed object shape, size or upright priors are used. The support plane is
estimated from the scene cloud; an explicitly labelled prior is only a
fallback. No ROS imports: callers supply points, mask and transforms, so the
module is testable offline. Truth poses are never an input.

Known limits recorded in the output: only the visible surface is observed,
so the hull misses hidden faces and the supported height assumes the object
rests on the estimated support plane.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import json

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from piperx_lab.handeye import as_transform

SOURCE = 'rgbd_object_segmentation'
BASE_FRAME = 'base_link'


class ObservationError(ValueError):
    """Raised when a generic observation cannot be produced or parsed."""


@dataclass
class GenericObjectObservation:
    """Shared observation structure for simulation and the real camera.

    ``dimensions_m`` is [d1, d2, height] in the reported OBB axes with the
    height taken from the estimated support plane to the topmost observed
    point. ``centroid_base_m`` is the visible-surface hull centroid and
    ``obb_center_base_m`` the supported box centre used for planning.
    """
    stamp: dict
    base_frame: str
    camera_frame: str
    joint_positions_rad: dict
    T_base_camera: list
    centroid_base_m: list
    dimensions_m: list
    obb_center_base_m: list
    obb_axes_base: list
    segmentation_confidence: float
    source: str = SOURCE
    status: str = 'ok'
    mask_path: str = ''
    points_base_path: str = ''
    convex_hull_path: str = ''
    quality: dict = field(default_factory=dict)
    symmetry: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            'status': self.status, 'source': self.source, 'stamp': self.stamp,
            'base_frame': self.base_frame, 'camera_frame': self.camera_frame,
            'joint_positions_rad': self.joint_positions_rad,
            'T_base_camera': self.T_base_camera,
            'mask_path': self.mask_path, 'points_base_path': self.points_base_path,
            'convex_hull_path': self.convex_hull_path,
            'centroid_base_m': self.centroid_base_m,
            'dimensions_m': self.dimensions_m,
            'obb_center_base_m': self.obb_center_base_m,
            'obb_axes_base': self.obb_axes_base,
            'segmentation_confidence': self.segmentation_confidence,
            'quality': self.quality, 'symmetry': self.symmetry,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ObservationError('Generic observation must be a JSON object')
        if data.get('status') != 'ok' or data.get('source') != SOURCE:
            raise ObservationError(
                f"Expected status ok with source {SOURCE}, got "
                f"{data.get('status')!r}/{data.get('source')!r}")
        if data.get('base_frame') != BASE_FRAME:
            raise ObservationError(f"Expected base frame {BASE_FRAME}")
        stamp = data.get('stamp')
        if not isinstance(stamp, dict) or {'sec', 'nanosec'} - set(stamp):
            raise ObservationError('Missing image sample stamp')
        center = cls._finite_vector(data.get('obb_center_base_m'), 3, 'obb_center_base_m')
        cls._finite_vector(data.get('centroid_base_m'), 3, 'centroid_base_m')
        dims = cls._finite_vector(data.get('dimensions_m'), 3, 'dimensions_m')
        if np.any(np.asarray(dims) <= 0):
            raise ObservationError('dimensions_m must be positive')
        axes = data.get('obb_axes_base')
        if (not isinstance(axes, list) or len(axes) != 3 or
                any(len(row) != 3 for row in axes)):
            raise ObservationError('obb_axes_base must be a 3x3 axis list')
        as_transform(np.asarray(data.get('T_base_camera'), dtype=float))
        return cls(
            stamp={'sec': int(stamp['sec']), 'nanosec': int(stamp['nanosec'])},
            base_frame=data['base_frame'], camera_frame=str(data.get('camera_frame', '')),
            joint_positions_rad=dict(data.get('joint_positions_rad') or {}),
            T_base_camera=np.asarray(data['T_base_camera'], dtype=float).tolist(),
            centroid_base_m=[float(v) for v in data['centroid_base_m']],
            dimensions_m=[float(v) for v in dims],
            obb_center_base_m=[float(v) for v in center],
            obb_axes_base=[[float(v) for v in row] for row in axes],
            segmentation_confidence=float(data.get('segmentation_confidence', 0.0)),
            mask_path=str(data.get('mask_path', '')),
            points_base_path=str(data.get('points_base_path', '')),
            convex_hull_path=str(data.get('convex_hull_path', '')),
            quality=dict(data.get('quality') or {}),
            symmetry=dict(data.get('symmetry') or {}),
            notes=list(data.get('notes') or []),
        )

    @staticmethod
    def _finite_vector(value, size, name):
        array = np.asarray(value, dtype=float)
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ObservationError(f'{name} must be {size} finite numbers')
        return array

    def center(self):
        """Planning centre; the supported box centre, not the visible centroid."""
        return list(self.obb_center_base_m)

    def supported_height_m(self):
        return float(self.dimensions_m[2])

    def vertical_axis_symmetry(self):
        return str((self.symmetry.get('vertical_axis') or {}).get('class', 'unknown'))

    def hold_check_mode(self):
        """Upright-tilt gating only where a tip-over is physically meaningful."""
        return 'upright_tilt' if self.vertical_axis_symmetry() == 'likely' else 'displacement_only'


def _finite_cloud(points, name, min_points):
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ObservationError(f'{name} must be an Nx3 point cloud')
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if len(cloud) < min_points:
        raise ObservationError(f'{name} has only {len(cloud)} finite points, need {min_points}')
    return cloud


def estimate_table_height(scene_base, z_low, z_high, bin_m=0.005,
                          min_peak_fraction=0.12, max_residual_m=0.008):
    """Dominant horizontal support plane from the scene-cloud z histogram.

    Returns (height_m, diagnostics). Raises ObservationError when no flat
    surface dominates the searched window.
    """
    zs = scene_base[:, 2]
    window = zs[(zs >= z_low) & (zs <= z_high)]
    if len(window) < 200:
        raise ObservationError('Too few scene points to estimate the support plane')
    edges = np.arange(z_low, z_high + bin_m, bin_m)
    hist, _ = np.histogram(window, bins=edges)
    peak = int(hist.argmax())
    center = (edges[peak] + edges[peak + 1]) / 2
    near = window[np.abs(window - center) <= 2 * bin_m]
    residual = float(np.median(np.abs(near - np.median(near))))
    strength = float(hist[peak]) / len(window)
    if strength < min_peak_fraction or residual > max_residual_m:
        raise ObservationError(
            f'No dominant support plane (peak fraction {strength:.2f}, residual {residual * 1e3:.1f} mm)')
    return float(np.median(near)), {'peak_fraction': strength, 'z_residual_m': residual}


def _connected_labels(points, radius):
    """Euclidean clustering through a cKDTree radius graph and union-find."""
    count = len(points)
    parent = np.arange(count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(points)
    pairs = tree.query_pairs(radius, output_type='ndarray')
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    labels = np.array([find(i) for i in range(count)])
    return labels, tree


def statistical_filter(points, k=12, std_ratio=2.0):
    """Remove points whose mean kNN distance is an outlier; keeps surfaces."""
    tree = cKDTree(points)
    k = min(k + 1, len(points))
    distances, _ = tree.query(points, k=k)
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = mean_distance.mean() + std_ratio * mean_distance.std()
    keep = mean_distance <= threshold
    return points[keep], float(keep.mean())


def _hull_centroid(vertices, simplices, interior):
    """Volume centroid of a closed hull from tetrahedra to an interior point."""
    v0 = vertices[simplices[:, 0]] - interior
    v1 = vertices[simplices[:, 1]] - interior
    v2 = vertices[simplices[:, 2]] - interior
    volumes = np.abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2))) / 6.0
    centroids = (vertices[simplices[:, 0]] + vertices[simplices[:, 1]] +
                 vertices[simplices[:, 2]] + interior) / 4.0
    return (volumes[:, None] * centroids).sum(axis=0) / volumes.sum()


def _side_band_circularity(points, axis_xy, table_z, top_z):
    """Plan-view circularity of thin height slices about the vertical axis.

    Within one thin z-slice, a rotationally symmetric body keeps every
    surface point at one radius (cylinder, sphere, cone all qualify), while
    a box sweeps from face to corner. The metric is the mean per-slice
    coefficient of variation, so spheres score circular despite their
    height-varying silhouette. Only side-band points are used.
    """
    height = top_z - table_z
    band = points[(points[:, 2] > table_z + 0.2 * height) &
                  (points[:, 2] < top_z - 0.03 * height)]
    if len(band) < 40:
        return None
    radial = np.linalg.norm(band[:, :2] - np.asarray(axis_xy), axis=1)
    mean_radius = float(np.median(radial))
    if mean_radius <= 1e-6:
        return None
    z_lo, z_hi = float(band[:, 2].min()), float(band[:, 2].max())
    thickness = max(0.004, (z_hi - z_lo) / 8)
    slice_cvs = []
    for center in np.linspace(z_lo + thickness, z_hi - thickness, 5):
        in_slice = np.abs(band[:, 2] - center) <= thickness / 2
        if int(in_slice.sum()) < 15:
            continue
        slice_radial = radial[in_slice]
        slice_mean = float(slice_radial.mean())
        if slice_mean <= 1e-6:
            continue
        slice_cvs.append(float(slice_radial.std() / slice_mean))
    if not slice_cvs:
        return None
    return float(np.mean(slice_cvs)), mean_radius, int(len(band))


def _symmetry_hypothesis(horizontal_dims, height, hull_ratio, points=None,
                         axis_xy=None, table_z=None, top_z=None):
    """Heuristic rotational-symmetry hypothesis; never a shape claim.

    Circular evidence comes from the side-band radius spread about the
    vertical axis; dimension equality alone is NOT treated as continuous
    rotational symmetry because a square box matches it too.
    """
    d_long, d_short = max(horizontal_dims), min(horizontal_dims)
    ratio = d_short / d_long if d_long > 0 else 0.0
    dims_equal = abs(d_long - d_short) <= 0.12 * max(d_long, 1e-9)
    hypothesis = {'hull_to_obb_volume_ratio': float(hull_ratio),
                  'height_to_width': float(height / max(d_long, 1e-9)),
                  'horizontal_dims_m': [float(d_long), float(d_short)]}
    circularity = None
    if points is not None and axis_xy is not None and table_z is not None and top_z is not None:
        circularity = _side_band_circularity(points, axis_xy, table_z, top_z)
    if circularity is not None:
        cv, mean_radius, band_points = circularity
        hypothesis['side_band_radial_cv'] = float(cv)
        hypothesis['side_band_mean_radius_m'] = float(mean_radius)
        hypothesis['side_band_points'] = band_points
        if cv < 0.05:
            if dims_equal:
                hypothesis['class'] = 'likely'
                hypothesis['discrete_orders_deg'] = None
                hypothesis['basis'] = (f'side-band radius spread {cv:.3f} at noise level '
                                       f'and equal horizontal dims')
            else:
                hypothesis['class'] = 'unknown'
                hypothesis['discrete_orders_deg'] = None
                hypothesis['basis'] = f'circular side band (cv {cv:.3f}) but unequal dims'
        elif cv >= 0.10:
            hypothesis['class'] = 'unlikely'
            if ratio > 0.85:
                hypothesis['discrete_orders_deg'] = [0, 90, 180, 270]
                hypothesis['basis'] = f'non-circular side band (cv {cv:.3f}), near-square footprint'
            elif ratio < 0.6:
                hypothesis['discrete_orders_deg'] = [0, 180]
                hypothesis['basis'] = f'non-circular side band (cv {cv:.3f}), bar-like footprint'
            else:
                hypothesis['discrete_orders_deg'] = None
                hypothesis['basis'] = f'non-circular side band (cv {cv:.3f}), asymmetric footprint'
        else:
            hypothesis['class'] = 'unknown'
            hypothesis['discrete_orders_deg'] = None
            hypothesis['basis'] = f'ambiguous side-band radius spread {cv:.3f}'
    else:
        # Too few side-band points for circularity; fall back to dims only,
        # and stay conservative about 'likely'.
        if dims_equal:
            hypothesis['class'] = 'unknown'
            hypothesis['discrete_orders_deg'] = None
            hypothesis['basis'] = (f'equal horizontal dims ({d_long * 1e3:.0f}/'
                                   f'{d_short * 1e3:.0f} mm) without circularity evidence; '
                                   'a square box would match too')
        elif ratio > 0.85:
            hypothesis['class'] = 'unlikely'
            hypothesis['discrete_orders_deg'] = [0, 90, 180, 270]
            hypothesis['basis'] = f'near-square footprint ({d_long * 1e3:.0f}/{d_short * 1e3:.0f} mm)'
        elif ratio < 0.6:
            hypothesis['class'] = 'unlikely'
            hypothesis['discrete_orders_deg'] = [0, 180]
            hypothesis['basis'] = f'bar-like footprint ({d_long * 1e3:.0f}/{d_short * 1e3:.0f} mm)'
        else:
            hypothesis['class'] = 'unknown'
            hypothesis['discrete_orders_deg'] = None
            hypothesis['basis'] = f'ambiguous footprint ratio {ratio:.2f}'
    hypothesis['note'] = 'heuristic from visible surface only; not a shape classification'
    return hypothesis


def build_generic_observation(mask_points_camera, scene_points_camera, T_base_camera,
                              *, stamp=None, camera_frame='', joint_positions_rad=None,
                              table_height_prior_m=None, cluster_radius_m=0.004,
                              roi_padding_m=0.05, roi_height_window_m=0.12,
                              min_cluster_points=200, min_mask_points=80,
                              sor_k=12, sor_std_ratio=2.0,
                              min_dimension_m=0.015, max_dimension_m=0.20,
                              min_height_m=0.015, max_height_m=0.25,
                              min_mask_overlap_fraction=0.5, min_hull_ratio=0.05):
    """Segment the masked target from the scene cloud and describe its geometry.

    ``mask_points_camera`` are the colour-mask surface points and
    ``scene_points_camera`` a sparse full-scene cloud, both in the camera
    optical frame; ``T_base_camera`` maps camera coordinates to the base
    frame. Returns ``{'status': 'ok', observation, artifacts}`` or
    ``{'status': 'rejected', 'reason'}``. Artifacts carry the numpy arrays
    for saving; the observation dict is JSON-safe once paths are filled.
    """
    T = as_transform(T_base_camera)
    try:
        mask_camera = _finite_cloud(mask_points_camera, 'mask points', min_mask_points)
        scene_camera = _finite_cloud(scene_points_camera, 'scene points', 500)
    except ObservationError as exc:
        return {'status': 'rejected', 'reason': str(exc)}
    mask_base = mask_camera @ T[:3, :3].T + T[:3, 3]
    scene_base = scene_camera @ T[:3, :3].T + T[:3, 3]

    anchor = np.median(mask_base, axis=0)
    try:
        table_z, table_stats = estimate_table_height(
            scene_base, anchor[2] - 0.30, anchor[2] + 0.02)
        table_source = 'scene_histogram'
    except ObservationError as exc:
        if table_height_prior_m is None:
            return {'status': 'rejected', 'reason': f'support plane: {exc}'}
        table_z = float(table_height_prior_m)
        table_source = 'prior_fallback'
        table_stats = {'peak_fraction': None, 'z_residual_m': None}
    if table_z > anchor[2] + 0.005:
        return {'status': 'rejected',
                'reason': f'estimated support plane {table_z:.3f} m is above the target surface'}

    # ROI around the mask keeps gripper or background clusters out.
    extent = np.ptp(mask_base[:, :2], axis=0)
    half = extent / 2 + roi_padding_m
    z_lo = table_z + 0.003
    z_hi = float(mask_base[:, 2].max()) + roi_height_window_m
    in_roi = ((np.abs(scene_base[:, 0] - anchor[0]) <= half[0]) &
              (np.abs(scene_base[:, 1] - anchor[1]) <= half[1]) &
              (scene_base[:, 2] >= z_lo) & (scene_base[:, 2] <= z_hi))
    roi_points = scene_base[in_roi]
    if len(roi_points) + len(mask_base) < min_cluster_points:
        return {'status': 'rejected',
                'reason': f'ROI contains {len(roi_points)} scene + {len(mask_base)} mask points, '
                          f'need {min_cluster_points}'}

    candidates = np.vstack((roi_points, mask_base))
    labels, _ = _connected_labels(candidates, cluster_radius_m)
    mask_start = len(roi_points)
    cluster_ids, counts = np.unique(labels[mask_start:], return_counts=True)
    if len(cluster_ids) == 0:
        return {'status': 'rejected', 'reason': 'mask points formed no cluster'}
    best = cluster_ids[counts.argmax()]
    object_mask = labels == best
    object_points = candidates[object_mask]
    mask_in_cluster = int(counts.max())
    overlap_fraction = mask_in_cluster / len(mask_base)
    if overlap_fraction < min_mask_overlap_fraction:
        return {'status': 'rejected',
                'reason': f'mask split across clusters ({mask_in_cluster}/{len(mask_base)} in the largest)'}
    cluster_count = int(np.unique(labels[:mask_start]).size)
    # Non-target clusters in the ROI are obstacles for planning (e.g. a
    # blocker or a second object); the selected target cluster is excluded.
    obstacle_boxes = []
    for label in np.unique(labels[:mask_start]):
        if label == best:
            continue
        cluster_points = candidates[labels == label]
        if len(cluster_points) < 40:
            continue
        low = cluster_points.min(axis=0)
        high = cluster_points.max(axis=0)
        if high[2] - low[2] < 0.015:
            continue  # flat remnant near the support plane
        obstacle_boxes.append({
            'aabb_min_m': [float(v) for v in low],
            'aabb_max_m': [float(v) for v in high],
            'points': int(len(cluster_points)),
        })
    obstacle_boxes.sort(key=lambda box: -box['points'])
    obstacle_boxes = obstacle_boxes[:3]

    filtered_points, inlier_fraction = statistical_filter(
        object_points, k=sor_k, std_ratio=sor_std_ratio)
    if len(filtered_points) < min_cluster_points:
        return {'status': 'rejected',
                'reason': f'statistical filter left {len(filtered_points)} of {len(object_points)} points'}

    roi_clipped = bool(np.any((np.abs(filtered_points[:, 0] - anchor[0]) > half[0] - 0.005) |
                              (np.abs(filtered_points[:, 1] - anchor[1]) > half[1] - 0.005)))

    try:
        hull = ConvexHull(filtered_points)
    except QhullError as exc:
        return {'status': 'rejected', 'reason': f'degenerate convex hull: {exc}'}
    visible_volume = float(hull.volume)
    # Only the visible surface was observed. A steep top-down view of a
    # faceted object can collapse to a near-planar cluster, so close the
    # solid by adding the cloud's shadow on the support plane. This is an
    # explicit rest-on-plane assumption, recorded in quality.
    shadow = filtered_points.copy()
    shadow[:, 2] = table_z
    completion = 'support_plane_shadow'
    try:
        hull = ConvexHull(np.vstack((filtered_points, shadow)))
        source_points = np.vstack((filtered_points, shadow))
    except QhullError:
        completion = None
        source_points = filtered_points
    # hull.simplices indexes the source cloud; remap to the hull vertex list.
    hull_indices = np.asarray(hull.vertices)
    index_map = {int(old): new for new, old in enumerate(hull_indices)}
    hull_vertices = source_points[hull_indices]
    hull_faces = np.array([[index_map[int(a)], index_map[int(b)], index_map[int(c)]]
                           for a, b, c in hull.simplices], dtype=np.int64)
    interior = source_points.mean(axis=0)
    centroid = _hull_centroid(hull_vertices, hull_faces, interior)

    # Gravity-aligned OBB: the up axis is exactly vertical in base_link, the
    # horizontal yaw comes from the PCA axes projected onto the plane.
    covariance = np.cov(filtered_points.T)
    _, eigenvectors = np.linalg.eigh(covariance)
    pca_axes = eigenvectors[:, ::-1].T  # descending variance
    pca_up = int(np.argmax(np.abs(pca_axes[:, 2])))
    pca_tilt_deg = float(math.degrees(math.acos(
        min(1.0, abs(float(pca_axes[pca_up, 2]))))))
    horizontal = []
    for index in (i for i in range(3) if i != pca_up):
        axis = pca_axes[index].copy()
        axis[2] = 0.0
        if np.linalg.norm(axis) > 1e-6:
            horizontal.append(axis)
    if len(horizontal) < 2:
        horizontal.append(np.array([1.0, 0.0, 0.0]))
    h1 = horizontal[0] / np.linalg.norm(horizontal[0])
    h2 = horizontal[1] - np.dot(horizontal[1], h1) * h1
    if np.linalg.norm(h2) < 1e-6:
        h2 = np.cross([0.0, 0.0, 1.0], h1)
    h2 /= np.linalg.norm(h2)
    up = np.array([0.0, 0.0, 1.0])
    obb_axes = np.array([h1, h2, up])
    projected = filtered_points @ obb_axes.T
    horizontal_extents = np.ptp(projected[:, :2], axis=0)
    supported_height = float(filtered_points[:, 2].max() - table_z)
    obb_volume = float(np.prod([*horizontal_extents, supported_height])) if supported_height > 0 else 0.0
    hull_ratio = float(hull.volume / obb_volume) if obb_volume > 0 else 0.0

    extents = [*horizontal_extents, supported_height]
    if np.any(np.asarray(extents[:2]) < min_dimension_m) or max(extents[:2]) > max_dimension_m:
        return {'status': 'rejected',
                'reason': f'horizontal extent {extents[:2]} outside [{min_dimension_m}, {max_dimension_m}] m'}
    if not min_height_m <= supported_height <= max_height_m:
        return {'status': 'rejected',
                'reason': f'supported height {supported_height:.3f} m outside '
                          f'[{min_height_m}, {max_height_m}] m'}
    if hull_ratio < min_hull_ratio:
        return {'status': 'rejected',
                'reason': f'hull fills only {hull_ratio:.3f} of its supported box '
                          f'(need >= {min_hull_ratio}); visible surface is too sparse'}

    horizontal_mid = (projected[:, :2].min(axis=0) + projected[:, :2].max(axis=0)) / 2
    obb_center = h1 * horizontal_mid[0] + h2 * horizontal_mid[1]
    obb_center[2] = table_z + supported_height / 2

    confidence = float(overlap_fraction * inlier_fraction)
    observation = {
        'status': 'ok', 'source': SOURCE, 'stamp': dict(stamp or {}),
        'base_frame': BASE_FRAME, 'camera_frame': camera_frame,
        'joint_positions_rad': dict(joint_positions_rad or {}),
        'T_base_camera': T.tolist(),
        'mask_path': '', 'points_base_path': '', 'convex_hull_path': '',
        'obstacles_base': obstacle_boxes,
        'centroid_base_m': [float(v) for v in centroid],
        'dimensions_m': [float(v) for v in extents],
        'obb_center_base_m': [float(v) for v in obb_center],
        'obb_axes_base': [[float(v) for v in axis] for axis in obb_axes],
        'segmentation_confidence': confidence,
        'quality': {
            'mask_points': int(len(mask_base)),
            'scene_points': int(len(scene_base)),
            'roi_points': int(len(roi_points)),
            'cluster_points': int(len(object_points)),
            'points_after_filter': int(len(filtered_points)),
            'cluster_count_in_roi': cluster_count,
            'obstacle_count': len(obstacle_boxes),
            'mask_overlap_fraction': float(overlap_fraction),
            'inlier_fraction': float(inlier_fraction),
            'table_height_m': float(table_z),
            'table_source': table_source,
            'table_peak_stats': table_stats,
            'supported_height_m': float(supported_height),
            'height_source': 'support_plane_to_top',
            'hull_volume_m3': float(hull.volume),
            'hull_visible_volume_m3': visible_volume,
            'hull_completion': completion,
            'obb_volume_m3': obb_volume,
            'hull_to_obb_volume_ratio': hull_ratio,
            'pca_vertical_tilt_deg': pca_tilt_deg,
            'aabb_extents_m': [float(v) for v in np.ptp(filtered_points, axis=0)],
            'hull_aabb_extents_m': [float(v) for v in np.ptp(hull_vertices, axis=0)],
            'roi_clipped': roi_clipped,
        },
        'symmetry': {'vertical_axis': _symmetry_hypothesis(
            horizontal_extents, supported_height, hull_ratio,
            points=filtered_points, axis_xy=centroid[:2],
            table_z=table_z, top_z=table_z + supported_height)},
        'notes': [
            'only the visible surface is observed; the hull is closed by a '
            'support-plane shadow when that helps (see quality.hull_completion)',
            'supported height assumes the object rests on the estimated support plane',
            'segmentation_confidence is a mask-overlap x inlier heuristic, not a probability',
        ],
    }
    artifacts = {
        'points_base': filtered_points,
        'points_camera': mask_camera,
        'hull_vertices': hull_vertices,
        'hull_faces': np.asarray(hull_faces, dtype=np.int64),
        'table_height_m': float(table_z),
    }
    return {'status': 'ok', 'observation': observation, 'artifacts': artifacts}


def write_ply(path, vertices, faces):
    """Write a minimal ASCII PLY with triangle faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ObservationError('PLY needs Nx3 vertices and Mx3 triangle faces')
    lines = ['ply', 'format ascii 1.0',
             f'element vertex {len(vertices)}',
             'property double x', 'property double y', 'property double z',
             f'element face {len(faces)}',
             'property list uchar int vertex_indices', 'end_header']
    lines += [f'{v[0]:.9f} {v[1]:.9f} {v[2]:.9f}' for v in vertices]
    lines += [f'3 {f[0]} {f[1]} {f[2]}' for f in faces]
    Path(path).write_text('\n'.join(lines) + '\n')


def read_ply(path):
    """Read vertices and triangle faces from the ASCII PLY written by write_ply."""
    lines = Path(path).read_text().splitlines()
    try:
        header_end = lines.index('end_header')
        counts = {}
        element = None
        for line in lines[:header_end]:
            parts = line.split()
            if parts[:1] == ['element']:
                element = parts[1]
                counts[element] = int(parts[2])
        vertex_count, face_count = counts['vertex'], counts['face']
        vertices = np.array([[float(v) for v in line.split()[:3]]
                             for line in lines[header_end + 1:header_end + 1 + vertex_count]])
        faces = np.array([[int(v) for v in line.split()[1:4]]
                          for line in lines[header_end + 1 + vertex_count:header_end + 1 + vertex_count + face_count]])
    except (ValueError, KeyError, IndexError) as exc:
        raise ObservationError(f'Unsupported or corrupted PLY {path}: {exc}') from exc
    if len(vertices) != vertex_count or len(faces) != face_count:
        raise ObservationError(f'PLY {path} body does not match its header')
    return vertices, faces


def save_generic_observation(out_dir, prefix, result):
    """Persist observation artifacts and fill the relative path fields.

    Writes ``{prefix}_object_points.npz``, ``{prefix}_object_hull.ply`` and
    ``{prefix}_mask.png``; the observation dict must already be built with
    status ok. Returns the observation with paths filled in.
    """
    if result.get('status') != 'ok':
        raise ObservationError('Only an ok result can be saved')
    observation = result['observation']
    artifacts = result['artifacts']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = f'{prefix}_object_points.npz'
    hull_path = f'{prefix}_object_hull.ply'
    mask_path = f'{prefix}_mask.png'
    np.savez_compressed(out_dir / points_path,
                        points_base=artifacts['points_base'],
                        points_camera=artifacts['points_camera'],
                        table_height_m=np.array(artifacts['table_height_m']))
    write_ply(out_dir / hull_path, artifacts['hull_vertices'], artifacts['hull_faces'])
    mask = artifacts.get('mask')
    if mask is not None:
        import cv2
        cv2.imwrite(str(out_dir / mask_path), (np.asarray(mask) * 255).astype(np.uint8))
    observation['points_base_path'] = points_path
    observation['convex_hull_path'] = hull_path
    observation['mask_path'] = mask_path if mask is not None else ''
    return observation


def load_object_points(path):
    """Load saved base-frame object points and the recorded support plane."""
    with np.load(Path(path), allow_pickle=False) as data:
        points = np.asarray(data['points_base'], dtype=np.float64)
        table = float(data['table_height_m'])
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ObservationError(f'Invalid saved object points in {path}')
    return points, table
