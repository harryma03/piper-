"""Colour-independent target seed selection from organized depth points.

This deliberately estimates a visible connected surface, not object identity
or a complete object mask. It cannot guarantee separation from touching surfaces.
"""
import cv2
import numpy as np


def select_depth_component(xyz, pixel, *, depth_tolerance_m=0.035,
                           radius_m=0.08, edge_tolerance_m=0.008, min_points=80,
                           max_fraction=0.30):
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError('Expected organized H x W x 3 point cloud')
    if len(pixel) != 2 or not np.isfinite(pixel).all():
        raise ValueError('Target pixel must contain finite x and y')
    x, y = (int(v) for v in pixel)
    height, width = points.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError('Target pixel is outside the image')
    valid = np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
    if not valid[y, x]:
        raise ValueError('所选像素没有有效深度，请点击目标内部')
    if not (0 < depth_tolerance_m <= 0.1 and 0 < radius_m <= 0.2
            and 0 < edge_tolerance_m <= 0.05):
        raise ValueError('Invalid depth selection bounds')
    seed = points[y, x]
    delta = points - seed
    nearby = (valid & (np.abs(delta[:, :, 2]) <= depth_tolerance_m) &
              (np.linalg.norm(delta, axis=2) <= radius_m))
    # Neighbouring image pixels must also have similar optical depth. A flat
    # depth window alone would merge adjacent objects at different depths.
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1] = (~nearby).astype(np.uint8)
    depth = np.nan_to_num(points[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | (2 << 8)
    cv2.floodFill(depth, flood_mask, (x, y), 0.0,
                  loDiff=edge_tolerance_m, upDiff=edge_tolerance_m, flags=flags)
    mask = flood_mask[1:-1, 1:-1] == 2
    count = int(mask.sum())
    if count < min_points:
        raise ValueError(f'选区有效点过少（{count}），请重新点击')
    if count > max_fraction * height * width:
        raise ValueError('选区过大，可能包含桌面或背景，请重新点击')
    return mask
