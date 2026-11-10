#!/usr/bin/env python3
"""Estimate an upright cylinder from isolated simulated RGB-D frames.

No target pose is read from Gazebo; no estimated coordinate topic is published.
Diameter, height, upright axis and support-plane height are task priors.
"""
import argparse
from collections import OrderedDict
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener, TransformException

from piperx_lab.camera_target import decode_color, decode_depth
from piperx_lab.cylinder_geometry import fit_cylinder
from piperx_lab.handeye import as_transform, transform_from_ros
from piperx_lab.rgbd_geometry import aligned_frames_valid, color_region_points


def observation_from_cloud(points_camera, T_base_camera, diameter_m=0.046,
                           height_m=0.08, table_height_m=-0.15):
    """Fit measured side points, then place full center using explicit priors."""
    fit = fit_cylinder(points_camera)
    if abs(fit['diameter_m'] - diameter_m) > 0.006:
        raise ValueError('Measured cylinder diameter disagrees with the known 46 mm object')
    T = as_transform(T_base_camera)
    axis = T[:3, :3] @ np.asarray(fit['axis_direction'])
    if abs(axis[2]) < math.cos(math.radians(15)):
        raise ValueError('Measured red-surface axis disagrees with the upright prior')
    ref = T[:3, :3] @ np.asarray(fit['axis_reference_xyz_m']) + T[:3, 3]
    center = [float(ref[0]), float(ref[1]), float(table_height_m + height_m / 2)]
    if not np.all(np.isfinite(center)):
        raise ValueError('Non-finite vision estimate')
    return {'source': 'sim_rgbd_red_cylinder_fit', 'base_frame': 'base_link',
            'center_xyz_m': center, 'axis_direction': [0., 0., 1.],
            'measured_red_axis_reference_base_m': ref.tolist(),
            'measured_axis_base': axis.tolist(), 'surface_fit': fit,
            'priors': {'diameter_m': diameter_m, 'height_m': height_m,
                       'table_height_m': table_height_m, 'upright': True},
            'independently_validated': False}


class SimObservation(Node):
    def __init__(self):
        super().__init__('piperx_sim_observation')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.frames = OrderedDict()
        self.depth_range = None
        self.result = None
        self.joints = None
        self.last_error = 'Waiting for synchronized RGB-D'
        self.buffer = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.buffer, self)
        for kind, typ, topic in [('color', Image, '/camera/color/aligned'),
                                 ('depth', Image, '/camera/depth'),
                                 ('info', CameraInfo, '/camera/info/depth')]:
            self.create_subscription(typ, topic, lambda m, k=kind: self.receive(k, m), 10)
        self.create_subscription(Float32, '/camera/depth_range', self.on_range, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)

    def on_joints(self, msg):
        if all(f'joint{i}' in msg.name for i in range(1, 7)):
            self.joints = msg

    def on_range(self, msg):
        if math.isfinite(msg.data) and msg.data > 0:
            self.depth_range = float(msg.data)

    def receive(self, kind, msg):
        if self.result is not None:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = self.frames.setdefault(stamp, {})
        bundle[kind] = msg
        while len(self.frames) > 8:
            self.frames.popitem(last=False)
        if not {'color', 'depth', 'info'} <= bundle.keys() or self.depth_range is None:
            return
        del self.frames[stamp]
        try:
            if self.joints is None:
                raise ValueError('No simulation joint feedback')
            jstamp = self.joints.header.stamp.sec + self.joints.header.stamp.nanosec * 1e-9
            if abs(stamp[0] + stamp[1] * 1e-9 - jstamp) > .2:
                raise ValueError('Camera/joint feedback not synchronized')
            if not aligned_frames_valid(bundle):
                raise ValueError('Unexpected RGB-D frame labels')
            now = self.get_clock().now().nanoseconds * 1e-9
            age = now - (stamp[0] + stamp[1] * 1e-9)
            if not math.isfinite(age) or abs(age) > 0.5:
                raise ValueError('Stale RGB-D frame')
            try:
                tf = self.buffer.lookup_transform('base_link', 'camera_depth_frame',
                                                  Time.from_msg(bundle['depth'].header.stamp))
                fallback = False
            except TransformException:
                tf = self.buffer.lookup_transform('base_link', 'camera_depth_frame', Time())
                fallback = True
            lag = abs(stamp[0] + stamp[1] * 1e-9 -
                      (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9))
            if lag > .2:
                raise ValueError(f'Camera/TF time difference {lag:.3f}s is too large')
            color = decode_color(bundle['color'])
            depth = decode_depth(bundle['depth'])
            points, mask = color_region_points(color, depth, bundle['info'], self.depth_range)
            obs = observation_from_cloud(points, transform_from_ros(tf))
            obs.update(stamp={'sec': stamp[0], 'nanosec': stamp[1]},
                       tf_lag_s=lag, tf_latest_fallback=fallback,
                       joint_positions_rad=dict(zip(self.joints.name, self.joints.position)))
            self.result = (obs, deepcopy(color), mask)
        except (ValueError, TransformException) as exc:
            self.last_error = str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=12.)
    args = parser.parse_args()
    if __import__('os').environ.get('ROS_DOMAIN_ID') != '183':
        raise SystemExit('Requires isolated simulation domain 183')
    rclpy.init()
    node = SimObservation()
    try:
        deadline = time.monotonic() + args.timeout
        while node.result is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        if node.result is None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({'status': 'failed', 'reason': node.last_error}, indent=2)+'\n')
            print(node.last_error, flush=True)
            return 1
        obs, color, mask = node.result
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'status': 'ok', **obs}, indent=2)+'\n')
        overlay = color.copy()
        ys, xs = np.nonzero(mask)
        cv2.rectangle(overlay, (int(xs.min()), int(ys.min())),
                      (int(xs.max()), int(ys.max())), (0, 255, 0), 2)
        cv2.imwrite(str(args.out.with_suffix('.png')), overlay)
        print('RGB-D observation saved', flush=True)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    raise SystemExit(main())
