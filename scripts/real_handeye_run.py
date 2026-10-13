#!/usr/bin/env python3
"""Drive the real Piper through varied poses and save hand-eye samples."""

import argparse
import math
import subprocess
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from piperx_lab.model import ARM_JOINTS, installed_urdf_path, read_joint_limits

# Joint-space offsets in degrees applied to the pose the run starts from. The
# camera sits on the wrist, so rotating joints 4 and 6 swings it around the
# tool axis and changes the viewing direction of the fixed board.
DELTAS_DEG = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (8.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-8.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 18.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -18.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 8.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, -8.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 25.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, -25.0),
    (6.0, 0.0, 0.0, 12.0, 4.0, 12.0),
    (-6.0, 0.0, 0.0, -12.0, -4.0, -12.0),
    (0.0, 0.0, 0.0, 20.0, 6.0, -20.0),
]


class PoseRunner(Node):
    def __init__(self):
        super().__init__('real_handeye_runner')
        self.pose = None
        self.pose_received_at = None
        self.create_subscription(JointState, '/feedback/joint_states', self._on_state, 10)
        self.action = ActionClient(self, FollowJointTrajectory,
                                   '/arm_controller/follow_joint_trajectory')

    def _on_state(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(name in values and math.isfinite(values[name]) for name in ARM_JOINTS):
            self.pose = [values[name] for name in ARM_JOINTS]
            self.pose_received_at = time.monotonic()

    def wait_pose(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.pose is not None and time.monotonic() - self.pose_received_at < 0.5:
                return list(self.pose)
        raise TimeoutError('no fresh joint state')

    def move(self, target, duration=5.0, steps=60):
        start = self.wait_pose()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        for index in range(1, steps + 1):
            u = index / steps
            blend = 0.5 - 0.5 * math.cos(math.pi * u)
            point = JointTrajectoryPoint()
            point.positions = [a + (b - a) * blend for a, b in zip(start, target)]
            stamp = round(duration * u * 1e9)
            point.time_from_start = Duration(sec=stamp // 1_000_000_000,
                                             nanosec=stamp % 1_000_000_000)
            goal.trajectory.points.append(point)
        if not self.action.wait_for_server(timeout_sec=10.0):
            raise TimeoutError('trajectory action unavailable')
        send = self.action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=15.0)
        if not send.done() or not send.result().accepted:
            raise RuntimeError('trajectory goal rejected')
        result = send.result().get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=duration + 30.0)
        if not result.done():
            raise TimeoutError('trajectory result timeout')
        outcome = result.result()
        if outcome.status != GoalStatus.STATUS_SUCCEEDED or outcome.result.error_code != 0:
            raise RuntimeError('trajectory failed: %s / %s'
                               % (outcome.status, outcome.result.error_string))


def capture(timeout):
    command = ['ros2', 'run', 'piperx_lab', 'handeye_capture', '--mode', 'real',
               '--timeout', str(timeout)]
    result = subprocess.run(command, text=True, capture_output=True)
    text = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode, (text[-1] if text else '')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Collect real Piper hand-eye samples')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--capture-timeout', type=float, default=15.0)
    parser.add_argument('--min-samples', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true',
                        help='Actually move the real arm; otherwise print the planned poses')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.duration) or args.duration <= 0
            or not math.isfinite(args.settle) or args.settle < 0
            or not math.isfinite(args.capture_timeout) or args.capture_timeout <= 0
            or args.min_samples < 1 or args.min_samples > len(DELTAS_DEG)):
        parser.error('duration and capture timeout must be positive, settle nonnegative, '
                     'and min-samples between 1 and 12')

    limits = read_joint_limits(installed_urdf_path())
    rclpy.init()
    node = PoseRunner()
    saved = 0
    try:
        base = node.wait_pose()
        print('start pose deg:', [round(math.degrees(v), 2) for v in base], flush=True)
        outside = [name for name, value in zip(ARM_JOINTS, base)
                   if not limits[name][0] <= value <= limits[name][1]]
        if outside:
            raise ValueError(f'current pose outside Piper URDF limits: {outside}')
        targets = []
        for delta in DELTAS_DEG:
            target = [v + math.radians(d) for v, d in zip(base, delta)]
            bad = [name for name, value in zip(ARM_JOINTS, target)
                   if not limits[name][0] <= value <= limits[name][1]]
            targets.append((delta, target, bad))
        if args.dry_run or not args.execute:
            for delta, _, bad in targets:
                print('delta', delta, 'limit issue' if bad else 'ok')
            if not args.execute:
                print('Preview only; add --execute to move the real arm')
            return 0
        for index, (delta, target, bad) in enumerate(targets, 1):
            if bad:
                print('%2d skip %s -> outside limits %s' % (index, delta, bad), flush=True)
                continue
            print('%2d moving %s' % (index, delta), flush=True)
            node.move(target, duration=args.duration)
            time.sleep(args.settle)
            code, message = capture(args.capture_timeout)
            print('   %s %s' % ('saved' if code == 0 else 'skip', message), flush=True)
            saved += code == 0
        print('returning to start pose', flush=True)
        node.move(base, duration=args.duration)
    except (TimeoutError, RuntimeError, ValueError) as exc:
        print('error: %s' % exc, file=sys.stderr)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print('captured %d/%d poses' % (saved, len(DELTAS_DEG)), flush=True)
    return 0 if saved >= args.min_samples else 1


if __name__ == '__main__':
    sys.exit(main())
