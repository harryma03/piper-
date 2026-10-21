"""Shared colour-mask to optical-frame point-cloud conversion."""

import math

import cv2
import numpy as np

from piperx_lab.camera_target import largest_color_mask
from piperx_lab.handeye import camera_model
from piperx_lab.transforms import TARGET_FRAME


def aligned_frames_valid(bundle):
    """Vidu labels its depth-aligned colour pixels with a separate frame alias."""
    return (bundle['depth'].header.frame_id == TARGET_FRAME and
            bundle['info'].header.frame_id == TARGET_FRAME and
            bundle['color'].header.frame_id in (TARGET_FRAME, 'camera_aligned_color_frame'))


def color_region_points(color, depth, info, range_m, target_color='red', min_area=80):
    if color.shape[:2] != depth.shape or depth.shape != (info.height, info.width):
        raise ValueError('Aligned RGB-D and CameraInfo dimensions disagree')
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError('Invalid depth range')
    mask = largest_color_mask(color, target_color, min_area)
    if mask is None:
        raise ValueError(f'No sufficiently large {target_color} region')
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    valid = mask & (depth > 0)
    if int(valid.sum()) < 80:
        raise ValueError('Too few valid depth pixels inside the colour region')
    ys, xs = np.nonzero(valid)
    pixels = np.column_stack((xs, ys)).astype(np.float64).reshape(-1, 1, 2)
    kind, k, d = camera_model(info)
    rays = (cv2.fisheye.undistortPoints(pixels, k, d) if kind == 'fisheye'
            else cv2.undistortPoints(pixels, k, d)).reshape(-1, 2)
    z = depth[valid].astype(np.float64) * range_m / 65535.0
    points = np.column_stack((rays[:, 0] * z, rays[:, 1] * z, z))
    if not np.all(np.isfinite(points)):
        raise ValueError('Colour point cloud contains invalid coordinates')
    return points, valid


def depth_points_camera(depth, info, range_m, stride=5):
    """Sparse metric depth cloud for local support-surface estimation."""
    if depth.shape != (info.height, info.width) or not math.isfinite(range_m) or range_m <= 0:
        raise ValueError('Invalid depth dimensions or range')
    ys, xs = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    valid = depth[ys, xs] > 0
    xs, ys = xs[valid], ys[valid]
    pixels = np.column_stack((xs, ys)).astype(np.float64).reshape(-1, 1, 2)
    if len(pixels) == 0:
        raise ValueError('No valid sparse depth points')
    kind, k, d = camera_model(info)
    rays = (cv2.fisheye.undistortPoints(pixels, k, d) if kind == 'fisheye'
            else cv2.undistortPoints(pixels, k, d)).reshape(-1, 2)
    z = depth[ys, xs].astype(np.float64) * range_m / 65535.0
    return np.column_stack((rays * z[:, None], z))
