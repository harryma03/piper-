"""Estimate an upright, known-size red cube from a synchronized RGB-D frame.

This is a read-only, task-specific perception baseline. A square cube's yaw is
only observable modulo 90 degrees; the published pose uses one representative.
"""

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener

from piperx_lab.camera_target import decode_color, decode_depth, largest_color_mask
from piperx_lab.handeye import camera_model, transform_from_ros
from piperx_lab.transforms import (BASE_FRAME, TARGET_FRAME, WRIST_FRAME,
                                   load_handeye, transform_points)
from piperx_lab.workspace import workspace_root


def cube_points_camera(color, depth, info, range_m, min_area=80):
    """Undistort all valid red pixels and return optical-frame XYZ in metres."""
    if color.shape[:2] != depth.shape or depth.shape != (info.height, info.width):
        raise ValueError('Aligned RGB-D and CameraInfo dimensions disagree')
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError('Invalid depth range')
    mask = largest_color_mask(color, 'red', min_area)
    if mask is None:
        raise ValueError('No red region large enough for a cube')
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    valid = mask & (depth > 0)
    if int(valid.sum()) < 80:
        raise ValueError('Too few valid depth pixels inside the red cube')
    ys, xs = np.nonzero(valid)
    pixels = np.column_stack((xs, ys)).astype(np.float64).reshape(-1, 1, 2)
    kind, k, d = camera_model(info)
    rays = (cv2.fisheye.undistortPoints(pixels, k, d) if kind == 'fisheye'
            else cv2.undistortPoints(pixels, k, d)).reshape(-1, 2)
    z = depth[valid].astype(np.float64) * range_m / 65535.0
    points = np.column_stack((rays[:, 0] * z, rays[:, 1] * z, z))
    if not np.all(np.isfinite(points)):
        raise ValueError('Red cube point cloud contains invalid coordinates')
    return points, valid


def estimate_upright_cube(points_base, size_m=0.04, table_height=None,
                          top_band_m=0.004):
    """Fit the visible top footprint of a square upright cube in base_link.

    Requires a substantial view of the top face. A partial side-only view is
    rejected instead of inventing an unobserved object centre or yaw.
    """
    points = np.asarray(points_base, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 80:
        raise ValueError('Need at least 80 observed cube points')
    if not np.all(np.isfinite(points)) or not math.isfinite(size_m) or size_m <= 0:
        raise ValueError('Invalid cube points or known size')
    top_z = float(np.quantile(points[:, 2], 0.98))
    top = points[np.abs(points[:, 2] - top_z) <= top_band_m]
    if len(top) < 60:
        raise ValueError('Cube top face has fewer than 60 depth points')
    rectangle = cv2.minAreaRect(top[:, :2].astype(np.float32))
    (cx, cy), (width, height), _ = rectangle
    spans = sorted((float(width), float(height)))
    if spans[0] < size_m * 0.60 or spans[1] > size_m * 1.40:
        raise ValueError(f'Incomplete or implausible cube top footprint: {spans}')
    if max(abs(span - size_m) for span in spans) > size_m * 0.30:
        raise ValueError(f'Observed top footprint differs from known cube size: {spans}')
    centre_z = top_z - size_m / 2
    if table_height is not None and abs((top_z - size_m) - table_height) > 0.008:
        raise ValueError('Cube bottom height disagrees with the configured table')
    corners = cv2.boxPoints(rectangle)
    edge = corners[1] - corners[0]
    yaw = (math.atan2(float(edge[1]), float(edge[0])) + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return {
        'center_xyz_m': [float(cx), float(cy), float(centre_z)],
        'top_center_xyz_m': [float(cx), float(cy), top_z],
        'yaw_mod_90_rad': yaw,
        'top_span_m': spans,
        'top_points': len(top),
        'observed_points': len(points),
        'size_assumed_m': size_m,
        'orientation_ambiguity_deg': 90,
        'position_kind': 'known_upright_cube_center',
    }


class CubePose(Node):
    def __init__(self, source, size_m, table_height, handeye, once, save_frame=None):
        super().__init__('piperx_cube_pose')
        self.source, self.size_m, self.table_height = source, size_m, table_height
        self.handeye, self.once = handeye, once
        self.save_frame = save_frame
        self.finished = False
        self.range_m = None
        self.frames = OrderedDict()
        self.last_report = 0.0
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, '/piperx/cube_pose_base', 10)
        self.top_pub = self.create_publisher(PointStamped, '/piperx/cube_top_base', 10)
        self.create_subscription(Image, '/camera/color/aligned',
                                 lambda msg: self._receive('color', msg), 10)
        self.create_subscription(Image, '/camera/depth',
                                 lambda msg: self._receive('depth', msg), 10)
        self.create_subscription(CameraInfo, '/camera/info/depth',
                                 lambda msg: self._receive('info', msg), 10)
        self.create_subscription(Float32, '/camera/depth_range', self._on_range, 10)

    def _on_range(self, msg):
        if math.isfinite(msg.data) and msg.data > 0:
            self.range_m = float(msg.data)

    def _receive(self, kind, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = self.frames.setdefault(key, {})
        bundle[kind] = msg
        while len(self.frames) > 6:
            self.frames.popitem(last=False)
        if not {'color', 'depth', 'info'} <= bundle.keys():
            return
        del self.frames[key]
        if self.range_m is None or time.monotonic() - self.last_report < 1.0:
            return
        self.last_report = time.monotonic()
        if any(message.header.frame_id != TARGET_FRAME for message in bundle.values()):
            self.get_logger().warn('Discarding RGB-D bundle with unexpected camera frame')
            return
        stamp = bundle['depth'].header.stamp
        age = self.get_clock().now().nanoseconds * 1e-9 - (stamp.sec + stamp.nanosec * 1e-9)
        if self.source == 'sim' and age < -0.5:
            # /clock can arrive after the first RGB-D frames on startup.
            return
        if not math.isfinite(age) or abs(age) > 0.5:
            self.get_logger().warn(f'Discarding stale cube RGB-D frame ({age:.3f} s)')
            return
        try:
            color = decode_color(bundle['color'])
            depth = decode_depth(bundle['depth'])
            points_camera, mask = cube_points_camera(color, depth, bundle['info'], self.range_m)
            target_frame = WRIST_FRAME if self.source == 'real' else TARGET_FRAME
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, target_frame,
                                                     Time.from_msg(stamp))
            except TransformException:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, target_frame, Time())
                lag = abs(stamp.sec + stamp.nanosec * 1e-9 -
                          (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9))
                if lag > 0.2:
                    raise ValueError(f'Camera/TF timestamp gap {lag:.3f} s exceeds 0.2 s')
            base_camera = transform_from_ros(tf)
            if self.source == 'real':
                base_camera = base_camera @ self.handeye['T_wrist_camera']
            result = estimate_upright_cube(transform_points(base_camera, points_camera),
                                           self.size_m, self.table_height)
        except (ValueError, TransformException) as exc:
            self.get_logger().warn(f'No valid cube pose: {exc}')
            return

        pose = PoseStamped()
        pose.header = bundle['depth'].header
        pose.header.frame_id = BASE_FRAME
        x, y, z = result['center_xyz_m']
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
        yaw = result['yaw_mod_90_rad']
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        self.pose_pub.publish(pose)
        top = PointStamped()
        top.header = pose.header
        top.point.x, top.point.y, top.point.z = result['top_center_xyz_m']
        self.top_pub.publish(top)
        if self.save_frame is not None:
            annotated = color.copy()
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
            centre = result['center_xyz_m']
            label = (f'cube centre: {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f} m; '
                     f'yaw {math.degrees(yaw):.1f} deg mod 90')
            cv2.rectangle(annotated, (5, 5), (color.shape[1] - 5, 46), (25, 35, 50), -1)
            cv2.putText(annotated, label, (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, (255, 255, 255), 1, cv2.LINE_AA)
            self.save_frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.save_frame), annotated):
                self.get_logger().error(f'Could not save cube pose frame: {self.save_frame}')
            else:
                result['annotated_frame'] = str(self.save_frame.resolve())
                self.save_frame = None
        print(json.dumps({'source': self.source, 'frame_id': BASE_FRAME,
                          'stamp': {'sec': stamp.sec, 'nanosec': stamp.nanosec},
                          **result}, ensure_ascii=False), flush=True)
        self.finished = self.once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('real', 'sim'), default='real')
    parser.add_argument('--size', type=float, default=0.04, help='Known cube edge in metres')
    parser.add_argument('--table-height', type=float,
                        help='Optional independently measured table Z in base_link metres')
    parser.add_argument('--handeye', type=Path)
    parser.add_argument('--save-frame', type=Path,
                        help='Save the first valid cube with mask and pose overlay')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--timeout', type=float, default=20.0)
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:] if argv is None else argv)
    if not math.isfinite(args.size) or args.size <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--size and --timeout must be positive finite values')
    if args.table_height is not None and not math.isfinite(args.table_height):
        parser.error('--table-height must be finite')
    handeye = None
    if args.source == 'real':
        path = args.handeye or workspace_root() / 'calibration/handeye_result.json'
        try:
            handeye = load_handeye(path)
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot use real hand-eye result: {exc}')
        if not handeye['quality_pass']:
            parser.error('Real hand-eye result did not pass its quality check')
    rclpy.init()
    node = CubePose(args.source, args.size, args.table_height, handeye, args.once,
                    args.save_frame)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.2)
            if args.once and time.monotonic() > deadline:
                print('Timed out waiting for a valid upright cube pose', file=sys.stderr)
                return 1
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        return 130
    finally:
        executor.shutdown()
        node.tf_listener.unregister()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
