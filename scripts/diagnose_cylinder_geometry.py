#!/usr/bin/env python3
"""Read-only C1W geometry diagnostic; stores local RGB-D evidence, no robot commands.

Run with the camera publishing. --table-roi x,y,w,h selects a clear tabletop
region in the aligned image; repeat for several regions. ROI selection must
be checked visually and is not an automatic support-surface detector.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Float32
from piperx_lab.camera_target import decode_color, decode_depth
from piperx_lab.cylinder_geometry import fit_cylinder
from piperx_lab.rgbd_geometry import color_region_points


def organized_sdk_cloud(msg, depth):
    if msg.is_bigendian:
        raise ValueError('Only the current little-endian C1W cloud format is supported')
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                      'offsets': [next(f.offset for f in msg.fields if f.name == a)
                                  for a in ['x', 'y', 'z']], 'itemsize': msg.point_step})
    raw = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                     strides=(msg.row_step, msg.point_step))
    packed = np.stack([raw[a] for a in ['x', 'y', 'z']], axis=-1).reshape(-1, 3)
    # Official wrapper skips Z==0 in row order; require a one-to-one map.
    if len(packed) != int((depth > 0).sum()):
        raise ValueError('SDK/depth valid counts differ; cannot map cloud to pixels')
    xyz = np.zeros((*depth.shape, 3), dtype=float)
    xyz[depth > 0] = packed
    return xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--known-diameter-m', type=float, help='Optional measured diameter prior for a separate diagnostic fit')
    parser.add_argument('--out', type=Path, default=Path('records/cylinder_real/geometry_diagnostic'))
    parser.add_argument('--table-roi', action='append', default=[], help='x,y,width,height of verified clear table')
    args = parser.parse_args()
    rois = [list(map(int, value.split(','))) for value in args.table_roi]
    rclpy.init()
    node = Node('c1w_geometry_diagnostic')
    frames, depth_range, captured = {}, [None], []

    def receive(kind, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = frames.setdefault(key, {})
        bundle[kind] = msg
        while len(frames) > 12:
            frames.pop(next(iter(frames)))
        if {'cloud', 'color', 'depth', 'info'} <= bundle.keys() and depth_range[0] is not None:
            age = node.get_clock().now().nanoseconds * 1e-9 - key[0] - key[1] * 1e-9
            if 0 <= age < .5:
                captured.append((key, bundle.copy()))

    for kind, typ, topic in [('cloud', PointCloud2, '/camera/pointcloud'),
                             ('color', Image, '/camera/color/aligned'),
                             ('depth', Image, '/camera/depth'),
                             ('info', CameraInfo, '/camera/info/depth')]:
        node.create_subscription(typ, topic, lambda m, k=kind: receive(k, m), 10)
    node.create_subscription(Float32, '/camera/depth_range', lambda m: depth_range.__setitem__(0, m.data), 10)
    try:
        end = time.monotonic() + 15
        while not captured and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.1)
        if not captured:
            raise TimeoutError('No fresh synchronized cloud, aligned color, depth and intrinsics')
        stamp, bundle = captured[0]
        color, depth = decode_color(bundle['color']), decode_depth(bundle['depth'])
        points, mask = color_region_points(color, depth, bundle['info'], depth_range[0])
        xyz = organized_sdk_cloud(bundle['cloud'], depth)
        native = xyz[mask]
        def checked_fit(cloud, known_diameter=None):
            try:
                return fit_cylinder(cloud, known_diameter_m=known_diameter)
            except ValueError as exc:
                return {'rejected': True, 'reason': str(exc)}
        free, sdk = checked_fit(points), checked_fit(native)
        axis = np.array(sdk['axis_direction']) if 'axis_direction' in sdk else None
        error = np.linalg.norm(points - native, axis=1)
        result = {'stamp': list(stamp), 'preview_only': True,
                  'reprojection_sdk_difference_m_quantiles_50_95': np.quantile(error, [.5, .95]).tolist(),
                  'reprojected_fit': free, 'sdk_fit': sdk, 'table_regions': [],
                  'scope': 'Camera-frame comparison; no handeye or robot TF used. Table ROIs require visual verification.'}
        if args.known_diameter_m is not None:
            result['measured_diameter_prior_fit'] = checked_fit(native, args.known_diameter_m)
            prior = result['measured_diameter_prior_fit']
            if 'axis_direction' in prior:
                result['prior_axis_note'] = 'Measured diameter is an input, not an independent size estimate'
        overlay = color.copy()
        overlay[mask] = [0, 0, 255]
        for roi in rois:
            if len(roi) != 4:
                raise ValueError('ROI must contain x,y,width,height')
            x, y, w, h = roi
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > depth.shape[1] or y+h > depth.shape[0]:
                raise ValueError('ROI exceeds image bounds')
            cloud = xyz[y:y+h, x:x+w].reshape(-1, 3)
            cloud = cloud[(cloud[:, 2] > 0) & np.all(np.isfinite(cloud), axis=1)]
            if len(cloud) < 80:
                raise ValueError('Insufficient valid table ROI points')
            center = cloud.mean(axis=0)
            _, _, vh = np.linalg.svd(cloud-center, full_matrices=False)
            normal = vh[-1]
            angle = float(np.degrees(np.arccos(np.clip(abs(normal @ axis), 0, 1)))) if axis is not None else None
            result['table_regions'].append({'roi_xywh': roi, 'points': len(cloud),
                'plane_normal_camera': normal.tolist(),
                'plane_rms_m': float(np.sqrt(np.mean(((cloud-center) @ normal)**2))),
                'cylinder_axis_normal_angle_deg': angle,
                'prior_axis_normal_angle_deg': float(np.degrees(np.arccos(np.clip(abs(normal @ np.array(result['measured_diameter_prior_fit']['axis_direction'])),0,1))))
                    if 'axis_direction' in result.get('measured_diameter_prior_fit', {}) else None})
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(overlay, f'{angle:.1f} deg' if angle is not None else 'axis rejected', (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out/'frame.npz', xyz=xyz, color=color, depth=depth, mask=mask,
                            k=np.array(bundle['info'].k), d=np.array(bundle['info'].d),
                            range_m=depth_range[0], stamp=np.array(stamp))
        cv2.imwrite(str(args.out/'regions.png'), overlay)
        (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
