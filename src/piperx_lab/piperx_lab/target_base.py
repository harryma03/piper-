"""Convert an Okulo camera-frame target into the Piper base frame.

Read-only: real mode requires a measured hand-eye result whose frames and
quality check pass; sim mode reads its own camera TF. Both publish a visible
surface point and an RViz marker, without commanding the arm.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.utilities import remove_ros_args
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from piperx_lab.handeye import transform_from_ros
from piperx_lab.transforms import (BASE_FRAME, WRIST_FRAME, TARGET_FRAME,
                                   camera_point_to_base, load_handeye, transform_points)
from piperx_lab.workspace import workspace_root


class TargetBase(Node):
    def __init__(self, handeye: dict | None, source: str, base_frame: str, wrist_frame: str,
                 out_topic: str, tf_timeout: float, max_tf_lag: float,
                 max_target_age: float, once: bool):
        super().__init__('piperx_target_base')
        self.handeye = handeye
        self.source = source
        self.base_frame = base_frame
        self.wrist_frame = wrist_frame
        self.tf_timeout = Duration(seconds=tf_timeout)
        self.max_tf_lag = max_tf_lag
        self.max_target_age = max_target_age
        self.once = once
        self.finished = False
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PointStamped, out_topic, 10)
        self.marker_pub = self.create_publisher(Marker, out_topic + '_marker', 10)
        self.create_subscription(PointStamped, '/piperx/target_camera',
                                 self._on_target, 10)

    def _on_target(self, msg):
        expected_camera = self.handeye['camera_frame'] if self.handeye else TARGET_FRAME
        if msg.header.frame_id != expected_camera:
            self.get_logger().error(
                f'Discarding target in {msg.header.frame_id!r}; expected '
                f'{expected_camera!r}')
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - self._to_seconds(msg.header.stamp)
        if not math.isfinite(age) or abs(age) > self.max_target_age:
            self.get_logger().warn(
                f'Discarding target with timestamp offset {age:.3f} s; '
                f'limit is {self.max_target_age:.3f} s')
            return
        camera = [msg.point.x, msg.point.y, msg.point.z]
        if not all(math.isfinite(value) for value in camera) or camera[2] <= 0:
            self.get_logger().error('Discarding invalid camera target')
            return
        try:
            target_frame = self.wrist_frame if self.source == 'real' else expected_camera
            tf, used_latest = self._lookup_transform(target_frame, msg.header.stamp)
        except TransformException as exc:
            self.get_logger().error(f'Cannot find {self.base_frame}->{target_frame} TF: {exc}')
            return
        base_target = transform_from_ros(tf)
        if self.source == 'real':
            point = camera_point_to_base(self.handeye['T_wrist_camera'], base_target, camera)
        else:
            point = transform_points(base_target, camera)

        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame
        out.point.x, out.point.y, out.point.z = (float(value) for value in point)
        self.pub.publish(out)
        self.marker_pub.publish(self._marker(out))
        print(json.dumps({
            'frame_id': self.base_frame,
            'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
            'camera_frame': msg.header.frame_id,
            'camera_xyz_m': [round(float(v), 4) for v in camera],
            'base_xyz_m': [round(float(v), 4) for v in point],
            'source': self.source,
            'tf_from_latest': used_latest,
        }, ensure_ascii=False), flush=True)
        self.finished = self.once

    @staticmethod
    def _to_seconds(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def _lookup_transform(self, target_frame, stamp):
        """Resolve base_link->target_frame, tolerating normal camera/TF stamp jitter.

        An exact-stamp lookup is attempted first; robot_state_publisher lags the
        camera by a few milliseconds, so a slightly stale latest transform is
        accepted when the lag stays within ``max_tf_lag`` seconds.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, target_frame, Time.from_msg(stamp), self.tf_timeout)
            return tf, False
        except TransformException:
            pass
        latest = self.tf_buffer.lookup_transform(self.base_frame, target_frame, Time())
        lag = abs(self._to_seconds(stamp) - self._to_seconds(latest.header.stamp))
        if lag > self.max_tf_lag:
            raise TransformException(
                f'latest {self.base_frame}->{target_frame} TF is {lag:.3f} s away from the '
                f'camera stamp; limit is {self.max_tf_lag:.3f} s')
        self.get_logger().warn(
            f'Using latest TF ({lag * 1000:.0f} ms behind the camera stamp)', once=True)
        return latest, True

    def _marker(self, point: PointStamped) -> Marker:
        marker = Marker()
        marker.header = point.header
        marker.ns = 'piperx_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 0.85
        return marker


def main(argv=None):
    parser = argparse.ArgumentParser(description='Publish the observed target in base_link')
    parser.add_argument('--handeye', type=Path,
                        help='Result file; defaults to <workspace>/calibration/handeye_result.json')
    parser.add_argument('--source', choices=('real', 'sim'), default='real',
                        help='Real uses measured hand-eye; sim uses its live camera TF')
    parser.add_argument('--base-frame', default=BASE_FRAME)
    parser.add_argument('--wrist-frame', default=WRIST_FRAME)
    parser.add_argument('--out-topic', default='/piperx/target_base')
    parser.add_argument('--once', action='store_true', help='Exit after the first converted target')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--tf-timeout', type=float, default=0.1,
                        help='Seconds to wait for an exact-stamp TF before falling back')
    parser.add_argument('--max-tf-lag', type=float, default=0.2,
                        help='Largest allowed camera-to-TF age when falling back to latest')
    parser.add_argument('--max-target-age', type=float, default=0.5,
                        help='Largest allowed target timestamp offset from current time')
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:] if argv is None else argv)
    if args.source == 'real' and args.handeye is None:
        try:
            args.handeye = workspace_root() / 'calibration/handeye_result.json'
        except FileNotFoundError as exc:
            parser.error(str(exc))
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive')
    if not math.isfinite(args.tf_timeout) or args.tf_timeout < 0:
        parser.error('--tf-timeout must be zero or positive')
    if not math.isfinite(args.max_tf_lag) or args.max_tf_lag <= 0:
        parser.error('--max-tf-lag must be positive')
    if not math.isfinite(args.max_target_age) or args.max_target_age <= 0:
        parser.error('--max-target-age must be positive')
    if args.base_frame != BASE_FRAME or args.wrist_frame != WRIST_FRAME:
        parser.error(f'Only {BASE_FRAME}->{WRIST_FRAME} is supported')
    handeye = None
    if args.source == 'real':
        try:
            handeye = load_handeye(args.handeye)
        except (OSError, ValueError) as exc:
            print(f'Hand-eye calibration unusable: {exc}', file=sys.stderr)
            return 1
        if not handeye['quality_pass']:
            print('Hand-eye calibration did not pass its quality check; refusing to '
                  'publish base-frame targets until it is re-solved', file=sys.stderr)
            return 1

    rclpy.init()
    node = TargetBase(handeye, args.source, args.base_frame, args.wrist_frame, args.out_topic,
                      args.tf_timeout, args.max_tf_lag, args.max_target_age, args.once)
    # TF subscriptions use a reentrant group, so they can fill the buffer while
    # a target callback waits for its exact-stamp transform.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.2)
            if args.once and time.monotonic() > deadline:
                print('Timed out waiting for a camera target', file=sys.stderr)
                return 1
    except (KeyboardInterrupt, ExternalShutdownException):
        return 130
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
