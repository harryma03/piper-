"""Record read-only 3D target repeatability and optional reference-point error.

The observed point represents the visible target surface. Keep the camera,
arm and target stationary when measuring repeatability; an object centre or
grasp pose requires a separate estimator.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args


FRAMES = {'camera': 'camera_depth_frame', 'base': 'base_link'}
MAX_TIMESTAMP_OFFSET_S = 0.5


def summarize_points(points, expected=None):
    """Population XYZ standard deviation and Euclidean errors, all in metres."""
    if not points:
        return None
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError('Expected finite XYZ samples')
    mean = xyz.mean(axis=0)
    summary = {
        'mean_xyz_m': mean.tolist(),
        'std_xyz_m': xyz.std(axis=0, ddof=0).tolist(),
        'std_definition': 'population, ddof=0',
        'max_distance_from_mean_m': float(np.linalg.norm(xyz - mean, axis=1).max()),
    }
    if expected is not None:
        reference = np.asarray(expected, dtype=np.float64)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError('Expected a finite reference XYZ')
        errors = np.linalg.norm(xyz - reference, axis=1)
        summary['reference_error'] = {
            'expected_xyz_m': reference.tolist(),
            'mean_error_xyz_m': (mean - reference).tolist(),
            'mean_point_error_m': float(np.linalg.norm(mean - reference)),
            'mean_sample_error_m': float(errors.mean()),
            'max_sample_error_m': float(errors.max()),
            'sample_rmse_m': float(np.sqrt(np.mean(errors ** 2))),
        }
    return summary


def sample_from_message(msg, now_ns: int, expected_frame: str, seen_stamps):
    """Validate a PointStamped without changing the accepted-stamp set."""
    if msg.header.frame_id != expected_frame:
        raise ValueError('wrong_frame')
    stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
    if stamp in seen_stamps:
        raise ValueError('duplicate_stamp')
    stamp_ns = stamp[0] * 1_000_000_000 + stamp[1]
    offset_s = (now_ns - stamp_ns) * 1e-9
    if abs(offset_s) > MAX_TIMESTAMP_OFFSET_S:
        raise ValueError('timestamp_offset')
    xyz = [float(msg.point.x), float(msg.point.y), float(msg.point.z)]
    if not all(math.isfinite(value) for value in xyz):
        raise ValueError('nonfinite_point')
    return {
        'stamp': {'sec': int(stamp[0]), 'nanosec': int(stamp[1])},
        'received_clock_ns': int(now_ns),
        'timestamp_offset_s': offset_s,
        'xyz_m': xyz,
    }


class TargetCheck(Node):
    def __init__(self, frame: str, count: int):
        super().__init__('piperx_target_check')
        self.expected_frame = FRAMES[frame]
        self.topic = '/piperx/target_' + frame
        self.requested_count = count
        self.samples = []
        self.seen_stamps = set()
        self.rejected = Counter()
        self.create_subscription(PointStamped, self.topic, self._on_target, 10)

    @property
    def finished(self):
        return len(self.samples) >= self.requested_count

    def _on_target(self, msg):
        if self.finished:
            return
        try:
            sample = sample_from_message(msg, self.get_clock().now().nanoseconds,
                                         self.expected_frame, self.seen_stamps)
        except ValueError as exc:
            self.rejected[str(exc)] += 1
            return
        self.seen_stamps.add((msg.header.stamp.sec, msg.header.stamp.nanosec))
        self.samples.append(sample)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', choices=tuple(FRAMES), default='camera')
    parser.add_argument('--source', choices=('real', 'sim'), default='real',
                        help='Report label only; select the matching ROS domain externally')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Wall-clock seconds allowed for collection')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', help='Description of the target and its measured position')
    parser.add_argument('--expected', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Expected visible surface point in the selected frame, metres')
    ros_args = sys.argv if argv is None else ['target_check', *argv]
    args = parser.parse_args(remove_ros_args(args=ros_args)[1:])
    if args.samples < 1:
        parser.error('--samples must be positive')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.expected is not None and not all(math.isfinite(v) for v in args.expected):
        parser.error('--expected must contain finite coordinates')

    rclpy.init(args=ros_args)
    node = TargetCheck(args.frame, args.samples)
    started = datetime.now(timezone.utc).isoformat()
    start_monotonic = time.monotonic()
    reason, exit_code = 'timeout', 1
    try:
        deadline = start_monotonic + args.timeout
        while rclpy.ok() and not node.finished and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.2, max(0.0, deadline - time.monotonic())))
        if node.finished:
            reason, exit_code = 'complete', 0
        elif not rclpy.ok():
            reason = 'shutdown'
    except KeyboardInterrupt:
        reason, exit_code = 'interrupted', 130
    except ExternalShutdownException:
        reason = 'shutdown'
    finally:
        report = {
            'schema_version': 1,
            'measurement': 'visible_surface_representative_point',
            'measurement_note': 'Visible surface representative point; not object centre or grasp pose. '
                                'Keep arm, camera and target stationary for repeatability checks.',
            'source': args.source,
            'label': args.label,
            'frame': args.frame,
            'frame_id': node.expected_frame,
            'topic': node.topic,
            'use_sim_time': bool(node.get_parameter('use_sim_time').value),
            'started_utc': started,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'duration_wall_s': time.monotonic() - start_monotonic,
            'requested_samples': args.samples,
            'accepted_samples': len(node.samples),
            'complete': node.finished,
            'stop_reason': reason,
            'max_timestamp_offset_s': MAX_TIMESTAMP_OFFSET_S,
            'rejected_messages': dict(node.rejected),
            'expected_xyz_m': args.expected,
            'statistics': summarize_points([sample['xyz_m'] for sample in node.samples],
                                           args.expected),
            'samples': node.samples,
        }
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                       allow_nan=False) + '\n', encoding='utf-8')
    except OSError as exc:
        print(f'Cannot save target check: {exc}', file=sys.stderr)
        return 1
    summary = {'complete': report['complete'], 'samples': len(report['samples']),
               'requested': args.samples, 'frame_id': report['frame_id'],
               'out': str(args.out.resolve()), 'statistics': report['statistics']}
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
