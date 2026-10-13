"""Inspect or command a short AgileX gripper width trajectory."""

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

from piperx_lab.model import installed_gripper_model_path, read_gripper_limit


class GripperMover(Node):
    def __init__(self, state_topic: str, action_name: str):
        super().__init__('piperx_gripper_move')
        self.width = None
        self.last_received = None
        self.create_subscription(JointState, state_topic, self._on_state, 10)
        self.action = ActionClient(self, FollowJointTrajectory, action_name)

    def _on_state(self, msg):
        values = dict(zip(msg.name, msg.position))
        width = values.get('gripper')
        if width is not None and math.isfinite(width):
            self.width = width
            self.last_received = time.monotonic()


def wait_for_state(node: GripperMover, timeout: float = 8.0) -> float:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.last_received is not None and time.monotonic() - node.last_received < 1.0:
            return node.width
    raise TimeoutError('No fresh gripper state received; check effector_type:=agx_gripper')


def check_width(current: float, target: float, limits: tuple[float, float], max_delta: float):
    if not math.isfinite(target):
        raise ValueError('Target width must be finite')
    low, high = limits
    if not low <= target <= high:
        raise ValueError(f'Width {target:.4f} m exceeds gripper limit [{low:.4f}, {high:.4f}] m')
    if abs(target - current) > max_delta:
        raise ValueError(f'Width changes {abs(target-current):.4f} m; limit is {max_delta:.4f} m')


def execute(node: GripperMover, width: float, duration: float):
    if not node.action.wait_for_server(timeout_sec=5.0):
        raise TimeoutError('Gripper trajectory action unavailable')
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ['gripper']
    point = JointTrajectoryPoint()
    point.positions = [width]
    duration_ns = round(duration * 1e9)
    point.time_from_start = Duration(
        sec=duration_ns // 1_000_000_000,
        nanosec=duration_ns % 1_000_000_000,
    )
    goal.trajectory.points = [point]
    send = node.action.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=8.0)
    if not send.done() or not send.result().accepted:
        raise RuntimeError('Gripper goal was not accepted')
    result = send.result().get_result_async()
    rclpy.spin_until_future_complete(node, result, timeout_sec=duration + 10)
    if not result.done():
        raise TimeoutError('Gripper trajectory result timed out')
    outcome = result.result()
    if outcome.status != GoalStatus.STATUS_SUCCEEDED or outcome.result.error_code != 0:
        raise RuntimeError(f'Gripper trajectory failed: status={outcome.status}, '
                           f'error={outcome.result.error_code} '
                           f'{outcome.result.error_string}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect or move AgileX gripper opening')
    parser.add_argument('--mode', choices=('sim', 'real'), required=True)
    parser.add_argument('--width', type=float, help='Total gripper opening, metres')
    parser.add_argument('--duration', type=float, default=2.0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--namespace', default='')
    args = parser.parse_args(argv)
    if args.execute and args.width is None:
        parser.error('--execute requires --width')
    if not math.isfinite(args.duration) or args.duration < 1.0:
        parser.error('--duration must be at least 1 second')
    limits = read_gripper_limit(installed_gripper_model_path())
    prefix = '/' + args.namespace.strip('/') if args.namespace.strip('/') else ''
    state_topic = prefix + ('/control/joint_states' if args.mode == 'sim' else '/feedback/joint_states')
    action_name = prefix + '/gripper_controller/follow_joint_trajectory'
    rclpy.init()
    node = GripperMover(state_topic, action_name)
    try:
        current = wait_for_state(node)
        print(f'Current gripper width: {current:.4f} m')
        if args.width is None:
            return 0
        check_width(current, args.width, limits, 0.025)
        print(f'Checked target width: {args.width:.4f} m')
        if not args.execute:
            print('Validation only; add --execute to send this trajectory')
            return 0
        execute(node, args.width, args.duration)
        print('Gripper trajectory action succeeded')
        return 0
    except (ValueError, TimeoutError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
