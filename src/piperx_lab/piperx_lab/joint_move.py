"""Inspect or command one short Piper arm trajectory through ros2_control."""

import argparse
import math
import sys
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from piperx_lab.model import ARM_JOINTS, installed_urdf_path, read_joint_limits


class JointMover(Node):
    def __init__(self, state_topic: str, action_name: str):
        super().__init__('piperx_joint_move')
        self.positions = None
        self.last_received = None
        self.create_subscription(JointState, state_topic, self._on_state, 10)
        self.action = ActionClient(self, FollowJointTrajectory, action_name)

    def _on_state(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(name in values and math.isfinite(values[name]) for name in ARM_JOINTS):
            self.positions = tuple(values[name] for name in ARM_JOINTS)
            self.last_received = time.monotonic()


def wait_for_state(node: JointMover, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.last_received is not None and time.monotonic() - node.last_received < 1.0:
            return node.positions
    raise TimeoutError('No fresh six-joint state received; check the selected mode and ROS topic')


def check_target(current, target, limits, max_delta):
    if len(target) != len(ARM_JOINTS) or not all(math.isfinite(v) for v in target):
        raise ValueError('Target must contain six finite joint angles in radians')
    for name, now, goal in zip(ARM_JOINTS, current, target):
        low, high = limits[name]
        if not math.isfinite(now) or not low - 0.01 <= now <= high + 0.01:
            raise ValueError(
                f'{name} current position {now:.4f} is outside Piper URDF limit '
                f'[{low:.4f}, {high:.4f}]; check joint zero and model before moving')
        if not low <= goal <= high:
            raise ValueError(f'{name} target {goal:.4f} exceeds Piper limit [{low:.4f}, {high:.4f}]')
        if abs(goal - now) > max_delta:
            raise ValueError(f'{name} changes {abs(goal-now):.4f} rad; limit is {max_delta:.4f} rad')


def execute(node: JointMover, target, duration):
    if not node.action.wait_for_server(timeout_sec=5.0):
        raise TimeoutError('Trajectory action unavailable')
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    point = JointTrajectoryPoint()
    point.positions = list(target)
    duration_ns = round(duration * 1e9)
    point.time_from_start = Duration(
        sec=duration_ns // 1_000_000_000,
        nanosec=duration_ns % 1_000_000_000,
    )
    goal.trajectory.points = [point]
    send = node.action.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=8.0)
    if not send.done() or not send.result().accepted:
        raise RuntimeError('Trajectory goal was not accepted')
    result = send.result().get_result_async()
    rclpy.spin_until_future_complete(node, result, timeout_sec=duration + 10)
    if not result.done():
        raise TimeoutError('Trajectory result timed out')
    outcome = result.result()
    if outcome.status != GoalStatus.STATUS_SUCCEEDED or outcome.result.error_code != 0:
        raise RuntimeError(f'Trajectory failed: status={outcome.status}, '
                           f'error={outcome.result.error_code} '
                           f'{outcome.result.error_string}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect or move Piper arm joints')
    parser.add_argument('--mode', choices=('sim', 'real'), required=True)
    parser.add_argument('--joints', nargs=6, type=float, metavar='RAD')
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--max-delta', type=float, default=0.15)
    parser.add_argument('--execute', action='store_true', help='Send the checked trajectory')
    parser.add_argument('--namespace', default='', help='ROS namespace, if one is in use')
    args = parser.parse_args(argv)
    if args.execute and args.joints is None:
        parser.error('--execute requires --joints')
    if not math.isfinite(args.duration) or args.duration < 2.0:
        parser.error('--duration must be at least 2 seconds')
    if not math.isfinite(args.max_delta) or not 0 < args.max_delta <= 0.15:
        parser.error('--max-delta must be in (0, 0.15] radians')

    limits = read_joint_limits(installed_urdf_path())
    prefix = '/' + args.namespace.strip('/') if args.namespace.strip('/') else ''
    state_topic = prefix + ('/control/joint_states' if args.mode == 'sim' else '/feedback/joint_states')
    action_name = prefix + '/arm_controller/follow_joint_trajectory'
    rclpy.init()
    node = JointMover(state_topic, action_name)
    try:
        current = wait_for_state(node)
        print('Current:', dict(zip(ARM_JOINTS, (round(v, 4) for v in current))))
        if args.joints is None:
            return 0
        check_target(current, args.joints, limits, args.max_delta)
        print('Checked target:', [round(v, 4) for v in args.joints])
        if not args.execute:
            print('Validation only; add --execute to send this trajectory')
            return 0
        execute(node, args.joints, args.duration)
        print('Trajectory action succeeded')
        return 0
    except (ValueError, TimeoutError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
