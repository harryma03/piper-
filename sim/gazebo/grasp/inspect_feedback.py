#!/usr/bin/env python3
"""Record simulation joint tracking and object pose after a grasp test."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetEntityState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action-log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=5.)
    args = parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '183':
        raise SystemExit('Requires isolated simulation domain 183')
    lines = [line for line in args.action_log.read_text().splitlines()
             if line.startswith('trajectory arm_controller ')]
    if not lines:
        raise ValueError('No arm trajectory in log')
    match = re.match(r'trajectory arm_controller (\[.*?\]) first \[.*?\] last (\[.*?\])', lines[-1])
    if match is None:
        raise ValueError('Could not parse target trajectory')
    names, values = ast.literal_eval(match.group(1)), ast.literal_eval(match.group(2))
    target = dict(zip(names, values))
    rclpy.init()
    node = Node('piperx_sim_feedback_inspector')
    sample = [None]
    node.create_subscription(JointState, '/joint_states', lambda m: sample.__setitem__(0, m), 10)
    client = node.create_client(GetEntityState, '/gazebo/get_entity_state')
    records = []
    try:
        end = time.monotonic() + args.seconds
        next_sample = 0.
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.1)
            now = time.monotonic()
            if sample[0] is not None and now >= next_sample:
                actual = dict(zip(sample[0].name, sample[0].position))
                records.append({'time_monotonic_s': now, 'stamp': {'sec': sample[0].header.stamp.sec,
                                 'nanosec': sample[0].header.stamp.nanosec},
                                'error_rad': {name: actual[name]-goal for name,goal in target.items()
                                              if name in actual}})
                next_sample = now + .5
        obj = None
        if client.wait_for_service(timeout_sec=3):
            future = client.call_async(GetEntityState.Request(name='cylinder', reference_frame='world'))
            rclpy.spin_until_future_complete(node, future, timeout_sec=3)
            if future.done() and future.result() and future.result().success:
                p = future.result().state.pose.position
                obj = [p.x, p.y, p.z]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'target_joint_positions_rad': target, 'samples': records,
                                        'object_xyz_m': obj}, indent=2)+'\n')
        print(json.dumps({'samples':len(records), 'last_error_rad':records[-1]['error_rad'] if records else None,
                          'object_xyz_m':obj}))
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
