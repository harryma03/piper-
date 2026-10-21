"""Geometry for cylinder grasp previews; no actuator or planning calls."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def model_tip_transform(model_path: Path):
    """Read the mean fingertip-frame origin relative to link6 from the official model."""
    root = ET.parse(model_path).getroot()
    def origin(name):
        joint = root.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f'Missing gripper model joint {name}')
        element = joint.find('origin')
        matrix = np.eye(4)
        matrix[:3, 3] = [float(v) for v in element.get('xyz', '0 0 0').split()]
        matrix[:3, :3] = Rotation.from_euler(
            'xyz', [float(v) for v in element.get('rpy', '0 0 0').split()]).as_matrix()
        return matrix
    base = origin('flange_joint') @ origin('gripper_base_joint')
    tip = np.eye(4)
    tip[:3, 3] = (origin('gripper_joint1')[:3, 3] +
                  origin('gripper_joint2')[:3, 3]) / 2
    return base @ tip


def grasp_candidates(base, link6_tip, link6_tcp, max_width=0.1,
                     clearance=0.01, approach_distance=0.10):
    reference = np.asarray(base['axis_reference_xyz_m'], np.float64)
    axis = np.asarray(base['axis_direction'], np.float64)
    diameter = float(base['diameter_m'])
    if (not np.all(np.isfinite(reference)) or not np.all(np.isfinite(axis)) or
            np.linalg.norm(axis) < 0.9 or not math.isfinite(diameter) or diameter <= 0):
        raise ValueError('Invalid metric cylinder observation')
    axis /= np.linalg.norm(axis)
    if axis[2] < 0:
        axis = -axis
    opening = diameter + 2 * clearance
    if opening > max_width:
        raise ValueError('Estimated cylinder plus clearance exceeds the gripper opening')
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    candidates = []
    tip_to_tcp = np.linalg.inv(link6_tip) @ link6_tcp
    for index in range(16):
        angle = ((index % 4) * math.pi / 4 if index < 4 else
                 (index % 4) * math.pi / 2 if index < 8 else (index - 8) * math.pi / 4)
        radial = math.cos(angle) * u + math.sin(angle) * v
        if index < 4:
            z = -axis
            y = radial  # Original gripper closes along its local Y axis.
            x = np.cross(y, z)
            kind = 'axial_approach'
        elif index < 8:
            z = -radial
            x = axis
            y = np.cross(z, x)
            kind = 'radial_approach'
        else:
            z = (-axis - radial) / math.sqrt(2)
            y = np.cross(axis, radial)
            x = np.cross(y, z)
            kind = 'oblique_approach'
        tip = np.eye(4)
        tip[:3, :3] = np.column_stack((x, y, z))
        offset = min(float(base.get('observed_red_span_m', 0)) * 0.2, 0.01) if index >= 8 else 0.0
        tip[:3, 3] = reference + axis * offset
        pre_tip = tip.copy()
        pre_tip[:3, 3] -= approach_distance * z
        candidates.append({'id': index, 'kind': kind, 'opening_m': opening,
                           'T_base_tip': tip.tolist(), 'T_base_tcp': (tip @ tip_to_tcp).tolist(),
                           'T_base_pregrasp_tcp': (pre_tip @ tip_to_tcp).tolist(),
                           'approach_direction': z.tolist(), 'preview_only': True,
                           'observed_band_axial_offset_m': offset,
                           'tip_reference': 'model_fingertip_frame_midpoint',
                           'contact_point_independently_validated': False})
    return candidates


def support_patch(points_base, reference):
    """Find a locally horizontal supporting surface below the observed red band."""
    points = np.asarray(points_base, np.float64)
    ref = np.asarray(reference, np.float64)
    valid = (np.all(np.isfinite(points), axis=1) &
             (np.linalg.norm(points[:, :2] - ref[:2], axis=1) < 0.30) &
             (points[:, 2] < ref[2] - 0.01) & (points[:, 2] > ref[2] - 0.25))
    points = points[valid]
    if len(points) < 120:
        raise ValueError('Not enough depth points below the cylinder to fit a support patch')
    rng = np.random.default_rng(21)
    if len(points) > 3000:
        points = points[rng.choice(len(points), 3000, replace=False)]
    best = None
    for _ in range(180):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if abs(normal[2]) < math.cos(math.radians(20)):
            continue
        mask = np.abs((points - a) @ normal) < 0.005
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < max(100, len(points) * 0.35):
        raise ValueError('No sufficiently supported local tabletop plane')
    inliers = points[best]
    centre = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centre, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((inliers - centre) @ normal) ** 2)))
    if rms > 0.004 or normal[2] < math.cos(math.radians(20)):
        raise ValueError('Support plane is too noisy or tilted for this preview')
    surface = ref - normal * float((ref - centre) @ normal)
    return {'surface_point_m': surface.tolist(), 'normal': normal.tolist(),
            'rms_m': rms, 'inliers': int(best.sum()), 'patch_size_m': [0.40, 0.40, 0.02],
            'coverage': 'local_support_patch_only'}
