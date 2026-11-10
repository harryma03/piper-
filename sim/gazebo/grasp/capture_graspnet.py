#!/usr/bin/env python3
"""Capture synchronized simulator RGB-D, image-time TF, joints and same-frame fit.

No Gazebo object pose is read. Requires isolated simulator ROS domain 183.

The default generic observation is shape- and size-agnostic: colour mask,
support-plane estimate, clustering, convex hull. The legacy cylinder and cube
branches remain as fixed-shape regression paths only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.time import Time
from tf2_ros import TransformException

from observe_cylinder import SimObservation, observation_from_cloud
from piperx_lab.camera_target import decode_color, decode_depth
from piperx_lab.handeye import camera_model, transform_from_ros
from piperx_lab.object_observation import build_generic_observation, save_generic_observation
from piperx_lab.rgbd_geometry import aligned_frames_valid, color_region_points


def cube_observation_from_cloud(points_camera, T_base_camera, edge_m=0.046,
                                table_height_m=-0.15):
    """Estimate an upright cube center from observed red surface points.

    Legacy fixed-size regression path; the generic observation replaces this.
    """
    points = np.asarray(points_camera, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
        raise ValueError('Insufficient red-cube surface points')
    T = np.asarray(T_base_camera, dtype=float)
    in_base = points @ T[:3, :3].T + T[:3, 3]
    low, high = np.quantile(in_base, [0.02, 0.98], axis=0)
    observed_xy = np.maximum(high[:2] - low[:2], 0.0)
    if np.any(observed_xy < 0.03) or np.any(observed_xy > 0.065):
        raise ValueError(f'Observed red-cube extents are inconsistent: {observed_xy.tolist()}')
    center = np.r_[(low[:2] + high[:2]) / 2, table_height_m + edge_m / 2]
    return {
        'source': 'sim_rgbd_red_cube_fit',
        'base_frame': 'base_link',
        'shape': 'cube',
        'dimensions_m': [edge_m, edge_m, edge_m],
        'center_xyz_m': center.tolist(),
        'axis_direction': [0.0, 0.0, 1.0],
        'observed_xy_extent_m': observed_xy.tolist(),
        'observed_surface_points': int(len(points)),
        'priors': {
            'edge_m': edge_m,
            'table_height_m': table_height_m,
            'upright': True,
        },
        'independently_validated': False,
    }


class Capture(SimObservation):
    def __init__(self, observation_mode, out_dir):
        super().__init__()
        self.observation_mode = observation_mode
        self.out_dir = Path(out_dir)
        self.joints_history = []
        # A camera frame may reach this process before its time-matched TF.
        # Keep the bundle and retry rather than substituting latest TF.
        self.create_timer(.02, self.retry_frames)

    def on_joints(self, msg):
        super().on_joints(msg)
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joints_history.append((stamp_s, msg))
        if len(self.joints_history) > 2000:
            del self.joints_history[:1000]

    def receive(self, kind, msg):
        if self.result is not None:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = self.frames.setdefault(stamp, {})
        bundle[kind] = msg
        while len(self.frames) > 32:
            self.frames.popitem(last=False)

    def retry_frames(self):
        if self.result is not None or self.depth_range is None:
            return
        # Newest frames first: recovery from a processing backlog should not
        # wait for stale bundles to fail their sync windows.
        for stamp, bundle in reversed(list(self.frames.items())):
            if {'color', 'depth', 'info'} <= bundle.keys():
                if self.try_bundle(stamp, bundle):
                    return

    def try_bundle(self, stamp, bundle):
        try:
            stamp_s = stamp[0] + stamp[1] * 1e-9
            # The image-time joint state is the history entry closest to the
            # camera stamp, not whatever arrived last; comparing against the
            # latest message breaks whenever processing falls behind.
            joints_msg = None
            if self.joints_history:
                _, joints_msg = min(self.joints_history,
                                    key=lambda item: abs(item[0] - stamp_s))
            if joints_msg is None:
                raise ValueError('No simulated joint feedback')
            joint_s = joints_msg.header.stamp.sec + joints_msg.header.stamp.nanosec * 1e-9
            if abs(stamp_s - joint_s) > .2:
                raise ValueError('Camera/joint feedback not synchronized')
            if not aligned_frames_valid(bundle):
                raise ValueError('Unexpected RGB-D frame labels')
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp_s
            if not math.isfinite(age) or abs(age) > .5:
                raise ValueError('Stale RGB-D frame')
            tf = self.buffer.lookup_transform('base_link', 'camera_depth_frame', Time.from_msg(bundle['depth'].header.stamp))
            fallback = False
            lag = abs(stamp_s - (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9))
            if lag > .2:
                raise ValueError(f'Camera/TF time difference {lag:.3f}s is too large')
            color, depth, info = decode_color(bundle['color']), decode_depth(bundle['depth']), bundle['info']
            target_points, mask = color_region_points(color, depth, info, self.depth_range)
            T = transform_from_ros(tf)
            joint_values = dict(zip(joints_msg.name, map(float, joints_msg.position)))
            stamp_obj = {'sec': int(stamp[0]), 'nanosec': int(stamp[1])}
            if self.observation_mode == 'generic':
                # Heavy generic construction runs in main() after capture;
                # doing it here would starve the joint-state subscriptions
                # and break the synchronization checks.
                obs = None
            elif self.observation_mode == 'cylinder':
                obs = observation_from_cloud(target_points, T)
            else:
                obs = cube_observation_from_cloud(target_points, T)
            if obs is not None:
                obs.update(stamp=stamp_obj, tf_lag_s=lag, tf_latest_fallback=fallback,
                           joint_positions_rad=joint_values)
            model, k, d = camera_model(info)
            yy, xx = np.indices(depth.shape, dtype=np.float32)
            pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
            if model == 'fisheye':
                rays = cv2.fisheye.undistortPoints(pixels, k, d).reshape(depth.shape + (2,))
            else:
                rays = cv2.undistortPoints(pixels, k, d).reshape(depth.shape + (2,))
            z = depth.astype(np.float32) * (self.depth_range / 65535.)
            xyz = np.concatenate((rays * z[..., None], z[..., None]), axis=-1).astype(np.float32)
            meta = {'status': 'ok', 'source': 'gazebo_simulated_rgbd',
                    'camera_frame': bundle['depth'].header.frame_id,
                    'base_frame': 'base_link', 'sample_stamp': stamp_obj,
                    'observation_mode': self.observation_mode,
                    'sample_joint_positions_rad': {f'joint{i}': joint_values[f'joint{i}'] for i in range(1, 7)},
                    'joint_stamp': {'sec': int(joints_msg.header.stamp.sec), 'nanosec': int(joints_msg.header.stamp.nanosec)},
                    'T_base_camera': T.tolist(), 'tf_lag_s': lag, 'tf_latest_fallback': fallback,
                    'depth_scale_m': self.depth_range / 65535., 'range_m': self.depth_range}
            self.result = {'xyz': xyz, 'color': color.copy(), 'depth': depth.copy(),
                           'mask': mask, 'k': k, 'd': d, 'stamp': stamp_obj, 'meta': meta,
                           'T_base_camera': T, 'target_points': target_points,
                           'joint_values': joint_values, 'obs': obs}
            return True
        except (ValueError, TransformException) as exc:
            self.last_error = str(exc)
            return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path, help='Output NPZ path; metadata and observation JSON are written alongside it')
    p.add_argument('--timeout', type=float, default=20.)
    p.add_argument('--observation-mode', choices=('generic', 'cylinder', 'cube'), default='generic',
                   help='generic: shape-agnostic pipeline; cylinder/cube: legacy fixed-shape regression fits')
    p.add_argument('--object-shape', choices=('cylinder', 'cube'), default=None,
                   help='Deprecated alias selecting a legacy observation mode')
    args = p.parse_args()
    if args.object_shape is not None:
        if args.observation_mode != 'generic':
            p.error('--object-shape and --observation-mode are mutually exclusive')
        args.observation_mode = args.object_shape
    if os.environ.get('ROS_DOMAIN_ID') != '183':
        raise SystemExit('Requires isolated simulation domain 183')
    rclpy.init()
    node = Capture(args.observation_mode, args.out.parent)
    try:
        deadline = time.monotonic() + args.timeout
        while node.result is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        if node.result is None:
            raise RuntimeError('Capture failed: ' + node.last_error)
        captured = node.result
        xyz, color, depth, mask = captured['xyz'], captured['color'], captured['depth'], captured['mask']
        k, d, stamp, meta = captured['k'], captured['d'], captured['stamp'], captured['meta']
        if args.observation_mode == 'generic':
            # Run the heavy shape-agnostic construction after capture, using
            # the same synchronized frame; no ROS traffic is needed here.
            valid_scene = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
            built = build_generic_observation(
                captured['target_points'], xyz[valid_scene][::2], captured['T_base_camera'],
                stamp=stamp, camera_frame=meta['camera_frame'],
                joint_positions_rad=captured['joint_values'])
            if built['status'] != 'ok':
                raise RuntimeError(f'Generic observation rejected: {built["reason"]}')
            built['artifacts']['mask'] = mask
            obs = save_generic_observation(args.out.parent, 'frame', built)
            obs.update(tf_lag_s=meta['tf_lag_s'], tf_latest_fallback=meta['tf_latest_fallback'])
        else:
            obs = captured['obs']
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, xyz=xyz, color=color, depth=depth, mask=mask, k=k, d=d,
                            stamp=np.array([stamp['sec'], stamp['nanosec']], dtype=np.int64))
        args.out.with_suffix('.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
        args.out.with_name(args.out.stem + '_observation.json').write_text(
            json.dumps(obs if obs.get('status') == 'ok' else {'status': 'ok', **obs},
                       indent=2, allow_nan=False) + '\n')
        cv2.imwrite(str(args.out.with_suffix('.png')), color)
        print(json.dumps({'npz': str(args.out), 'metadata': str(args.out.with_suffix('.json')),
                          'observation': str(args.out.with_name(args.out.stem + '_observation.json')),
                          'observation_mode': args.observation_mode,
                          'stamp': stamp, 'tf_lag_s': meta['tf_lag_s'], 'tf_latest_fallback': meta['tf_latest_fallback']}))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    raise SystemExit(main())
