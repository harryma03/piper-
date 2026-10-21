"""Fit an unknown circular cylinder to a visible curved surface.

The axial reference is the middle of the observed colour band, not the centre
of the complete object. Neither hidden end planes nor rotation about the axis
are observable from this fit.
"""

import math

import numpy as np
from scipy.optimize import least_squares


def fit_cylinder(points, known_diameter_m=None):
    known_radius = None
    if known_diameter_m is not None:
        if not math.isfinite(known_diameter_m) or not 0.002 < known_diameter_m < 0.3:
            raise ValueError('Invalid known cylinder diameter')
        known_radius = known_diameter_m / 2
    cloud = np.asarray(points, np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 80:
        raise ValueError('Need at least 80 valid red surface points')
    if not np.all(np.isfinite(cloud)):
        raise ValueError('Cylinder cloud must be finite')
    if len(cloud) > 2000:
        cloud = cloud[np.linspace(0, len(cloud) - 1, 2000).astype(int)]
    origin = np.median(cloud, axis=0)
    local = cloud - origin
    extent = float(np.linalg.norm(np.ptp(local, axis=0)))
    if extent < 0.01:
        raise ValueError('Observed region is too small for an unconstrained cylinder fit')
    _, _, axes = np.linalg.svd(local - local.mean(axis=0), full_matrices=False)
    radius_max = max(0.15, extent * 4)

    def direction(v):
        return np.array([math.cos(v[4]) * math.cos(v[3]),
                         math.cos(v[4]) * math.sin(v[3]), math.sin(v[4])])

    def residual(v):
        axis = direction(v)
        offsets = local - v[:3]
        radial = offsets - np.outer(offsets @ axis, axis)
        return np.r_[np.linalg.norm(radial, axis=1) - v[5], np.dot(v[:3], axis)]

    solutions = []
    for axis in axes:
        initial = np.r_[np.zeros(3), math.atan2(axis[1], axis[0]),
                        math.asin(float(np.clip(axis[2], -0.999999, 0.999999))),
                        min(extent / 3, radius_max / 2)]
        if known_radius is not None:
            initial[5] = known_radius
        radius_low = .001 if known_radius is None else known_radius - 1e-9
        radius_high = radius_max if known_radius is None else known_radius + 1e-9
        fit = least_squares(
            residual, initial,
            bounds=(np.r_[np.full(3, -radius_max * 2), -np.inf, -math.pi / 2, radius_low],
                    np.r_[np.full(3, radius_max * 2), np.inf, math.pi / 2, radius_high]),
            loss='soft_l1', f_scale=0.002, max_nfev=250)
        errors = residual(fit.x)[:-1]
        solutions.append((float(np.median(np.abs(errors))), fit.x, errors))
    _, parameters, errors = min(solutions, key=lambda item: item[0])
    axis = direction(parameters)
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    radius = float(parameters[5])
    threshold = max(0.003, 3 * float(np.median(np.abs(errors))))
    inliers = np.abs(errors) <= threshold
    if float(inliers.mean()) < 0.75:
        raise ValueError('Too many red points disagree with a circular cylinder')
    rmse = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    if rmse > max(0.0015, min(0.004, radius * 0.15)):
        raise ValueError('Cylinder residual is too large for a reliable measurement')
    offsets = local[inliers] - parameters[:3]
    axial = offsets @ axis
    lower, upper = np.quantile(axial, [0.02, 0.98])
    observed_length = float(upper - lower)
    radial = offsets - np.outer(axial, axis)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    basis_x = np.cross(axis, helper)
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(axis, basis_x)
    angles = np.sort(np.arctan2(radial @ basis_y, radial @ basis_x))
    arc = 2 * math.pi - float(np.max(np.diff(np.r_[angles, angles[0] + 2 * math.pi])))
    if arc < math.radians(50) or radius > radius_max * 0.95:
        raise ValueError('Insufficient visible curvature; diameter is not observable')
    if observed_length < max(0.01, radius * 0.5):
        raise ValueError('Too little axial extent to determine the cylinder axis')
    reference = origin + parameters[:3] + axis * float((lower + upper) / 2)
    return {
        'model': 'circular_cylinder_hypothesis',
        'diameter_source': 'measured_prior' if known_radius is not None else 'surface_fit',
        'axis_reference_xyz_m': reference.tolist(),
        'axis_direction': axis.tolist(),
        'diameter_m': radius * 2,
        'observed_red_span_m': observed_length,
        'surface_rmse_m': rmse,
        'visible_arc_deg': math.degrees(arc),
        'inlier_fraction': float(inliers.mean()),
        'points_used': len(cloud),
        'axis_sign_arbitrary': True,
        'rotation_about_axis_observable': False,
        'full_height_m': None,
        'object_center_xyz_m': None,
        'reference_kind': 'observed_red_band_axis_midpoint',
    }
