#!/usr/bin/env python3
"""Exercise the official MoveIt fake-hardware demo in ROS_DOMAIN_ID=185."""

import json
import os
from pathlib import Path
import time

if os.environ.get("ROS_DOMAIN_ID") != "185":
    raise SystemExit("This check requires the isolated ROS_DOMAIN_ID=185")

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from controller_manager_msgs.srv import ListHardwareComponents
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetMotionPlan
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "records/official_demo_tests/moveit_fake_result.json"
PREFIX = "/official_demo"


def main():
    rclpy.init()
    node = Node("official_moveit_demo_check")
    result = {"status": "running", "ros_domain_id": 185, "namespace": PREFIX, "stages": []}
    state = [None]
    subscription = node.create_subscription(
        JointState, PREFIX + "/control/joint_states", lambda msg: state.__setitem__(0, msg), 10)

    def wait(future, seconds=20):
        rclpy.spin_until_future_complete(node, future, timeout_sec=seconds)
        if not future.done():
            raise TimeoutError("ROS request timed out")
        return future.result()

    def state_after(names, targets, tolerance=0.002):
        # Require a new state received after the action result.
        state[0] = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if state[0] is not None:
                values = dict(zip(state[0].name, state[0].position))
                if all(n in values and abs(values[n] - v) < tolerance for n, v in zip(names, targets)):
                    return values
        raise RuntimeError("Fresh simulated joint feedback does not match target")

    def execute_action(client, goal):
        if not client.wait_for_server(timeout_sec=10):
            raise TimeoutError("Action server unavailable")
        handle = wait(client.send_goal_async(goal))
        if not handle.accepted:
            raise RuntimeError("Action rejected")
        response = wait(handle.get_result_async(), 30)
        if response.status != 4:
            raise RuntimeError(f"Action failed, status={response.status}")
        return response.result

    try:
        hardware = node.create_client(ListHardwareComponents, PREFIX + "/controller_manager/list_hardware_components")
        if not hardware.wait_for_service(timeout_sec=10):
            raise TimeoutError("Fake hardware service unavailable")
        components = wait(hardware.call_async(ListHardwareComponents.Request())).component
        details = [{"name": c.name, "plugin": c.class_type} for c in components]
        if not details or any(c["plugin"] != "mock_components/GenericSystem" for c in details):
            raise RuntimeError(f"Expected only mock hardware, found {details}")
        result["hardware"] = details

        deadline = time.monotonic() + 10
        while state[0] is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if state[0] is None:
            raise TimeoutError("No simulated joint states")

        names = [f"joint{i}" for i in range(1, 7)]
        values = dict(zip(state[0].name, state[0].position))
        target = [values[n] for n in names]
        target[0] += 0.1
        planner = node.create_client(GetMotionPlan, PREFIX + "/plan_kinematic_path")
        if not planner.wait_for_service(timeout_sec=10):
            raise TimeoutError("MoveIt planning service unavailable")
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "arm"
        motion.allowed_planning_time = 5.0
        motion.num_planning_attempts = 3
        motion.max_velocity_scaling_factor = 0.1
        motion.max_acceleration_scaling_factor = 0.1
        motion.start_state.joint_state = state[0]
        constraints = Constraints()
        for name, position in zip(names, target):
            constraints.joint_constraints.append(JointConstraint(
                joint_name=name, position=position, tolerance_above=0.001,
                tolerance_below=0.001, weight=1.0))
        motion.goal_constraints = [constraints]
        plan = wait(planner.call_async(request)).motion_plan_response
        if plan.error_code.val != 1 or not plan.trajectory.joint_trajectory.points:
            raise RuntimeError(f"MoveIt plan failed: {plan.error_code.val}")
        result["stages"].append({"stage": "arm_plan", "error_code": plan.error_code.val,
                                  "trajectory_points": len(plan.trajectory.joint_trajectory.points)})
        arm = ActionClient(node, ExecuteTrajectory, PREFIX + "/execute_trajectory")
        outcome = execute_action(arm, ExecuteTrajectory.Goal(trajectory=plan.trajectory))
        if outcome.error_code.val != 1:
            raise RuntimeError(f"MoveIt execution failed: {outcome.error_code.val}")
        feedback = state_after(names, target)
        result["stages"].append({"stage": "arm_execute", "error_code": outcome.error_code.val,
                                  "target": target, "feedback": {n: feedback[n] for n in names}})

        gripper = ActionClient(node, FollowJointTrajectory, PREFIX + "/gripper_controller/follow_joint_trajectory")
        for width in (0.04, 0.0):
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = ["gripper"]
            point = JointTrajectoryPoint(positions=[width])
            point.time_from_start.sec = 2
            goal.trajectory.points = [point]
            outcome = execute_action(gripper, goal)
            if outcome.error_code != 0:
                raise RuntimeError(f"Gripper action failed: {outcome.error_code}")
            feedback = state_after(["gripper"], [width])
            result["stages"].append({"stage": "gripper", "target_m": width,
                                      "feedback_m": feedback["gripper"], "error_code": outcome.error_code})
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
