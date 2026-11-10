#!/usr/bin/env python3
"""Independent scene-side disturbance injector (truth domain only).

Moves the simulated object with the Gazebo set_entity_state service on a
schedule. Runs as its own process with NO connection to the perception or
decision pipeline: the executor can only notice a disturbance through its
camera, which is exactly the experiment's premise. Domain 183 only.
"""
import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState


def main():
    parser = __import__('argparse').ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', required=True,
                        help='JSON list of {"delay_s": float, "shift_m": [dx, dy, dz]}')
    parser.add_argument('--object', default='cylinder')
    args = parser.parse_args()
    schedule = json.loads(args.schedule)
    if not isinstance(schedule, list) or not schedule:
        raise SystemExit('schedule must be a non-empty JSON list')
    rclpy.init()
    node = Node('disturbance_injector')
    client = node.create_client(SetEntityState, '/gazebo/set_entity_state')
    if not client.wait_for_service(timeout_sec=20):
        raise SystemExit('set_entity_state unavailable')
    print(json.dumps({'injector': 'ready', 'events': len(schedule)}), flush=True)
    start = time.monotonic()
    for event in schedule:
        delay = float(event['delay_s']) - (time.monotonic() - start)
        if delay > 0:
            time.sleep(delay)
        state_req = SetEntityState.Request()
        state_req.state.name = args.object
        state_req.state.pose.position.x = float(event['shift_m'][0])
        state_req.state.pose.position.y = float(event['shift_m'][1])
        state_req.state.pose.position.z = float(event['shift_m'][2])
        state_req.state.pose.orientation.w = 1.0
        # set_entity_state takes an absolute pose; the injector reads the
        # current state first and applies the shift as a relative offset.
        get_client = node.create_client(__import__(
            'gazebo_msgs.srv', fromlist=['GetEntityState']).GetEntityState,
            '/gazebo/get_entity_state')
        if not get_client.wait_for_service(timeout_sec=5):
            print(json.dumps({'event': 'failed', 'error': 'get_entity_state missing'}),
                  flush=True)
            continue
        current = get_client.call_async(__import__(
            'gazebo_msgs.srv', fromlist=['GetEntityState']).GetEntityState.Request(
                name=args.object, reference_frame='world'))
        rclpy.spin_until_future_complete(node, current, timeout_sec=5)
        if current.result() is None or not current.result().success:
            print(json.dumps({'event': 'failed', 'error': 'no current state'}),
                  flush=True)
            continue
        pose = current.result().state.pose
        state_req.state.pose.position.x = pose.position.x + float(event['shift_m'][0])
        state_req.state.pose.position.y = pose.position.y + float(event['shift_m'][1])
        state_req.state.pose.position.z = max(
            pose.position.z + float(event['shift_m'][2]), -0.145)
        state_req.state.pose.orientation = pose.orientation
        state_req.state.twist = current.result().state.twist
        future = client.call_async(state_req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        applied = future.result() is not None and future.result().success
        print(json.dumps({'event': 'shifted',
                          'at_s': round(time.monotonic() - start, 2),
                          'shift_m': event['shift_m'], 'applied': applied}),
              flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
