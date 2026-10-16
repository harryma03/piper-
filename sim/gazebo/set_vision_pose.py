"""Put only the Gazebo Piper in a pose facing the red test cube."""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


POSE = (0.0, 1.8, -1.65, 0.0, 0.0, 0.0)
JOINTS = tuple(f'joint{i}' for i in range(1, 7))


def main():
    rclpy.init()
    node = Node('piperx_gazebo_vision_pose')
    publisher = node.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)
    state = {'ready': False}

    def on_joint_state(msg: JointState):
        values = dict(zip(msg.name, msg.position))
        if all(name in values for name in JOINTS):
            state['ready'] = all(math.isclose(values[name], goal, abs_tol=0.03)
                                 for name, goal in zip(JOINTS, POSE))

    node.create_subscription(JointState, '/control/joint_states', on_joint_state, 10)
    deadline = time.monotonic() + 15.0
    try:
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if publisher.get_subscription_count() == 0:
            raise TimeoutError('Gazebo trajectory plugin is unavailable')
        command = JointTrajectory()
        command.header.frame_id = 'base_link'
        command.joint_names = list(JOINTS)
        point = JointTrajectoryPoint()
        point.positions = list(POSE)
        # Gazebo's joint-pose plugin applies this single point at once.
        point.time_from_start.sec = 0
        command.points = [point]
        publisher.publish(command)
        while not state['ready'] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not state['ready']:
            raise TimeoutError('Gazebo arm did not reach the vision test pose')
        print('Gazebo Piper reached red-target vision pose', flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
