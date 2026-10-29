"""Known-pose Gazebo contact test, domain 183 only. No attachment/teleporting."""

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from pathlib import Path
import time
from copy import deepcopy
import numpy as np
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    RobotState,
    AllowedCollisionEntry,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK, GetMotionPlan, GetCartesianPath
from gazebo_msgs.srv import GetEntityState
from gazebo_msgs.msg import ContactsState
from shape_msgs.msg import SolidPrimitive
from piperx_lab.cylinder_preview import CylinderPreview, pose_from_matrix
from piperx_lab.cylinder_grasp import grasp_candidates, model_tip_transform
from piperx_lab.model import installed_gripper_model_path

# These are target widths, not measured contact forces.
OPEN_WIDTH_M = 0.070
CLOSED_WIDTH_M = 0.032
GRIPPER_JOINTS = ("gripper", "gripper_joint1", "gripper_joint2")


def both_fingers_in_both_phases(contacts):
    """Require evidence from each finger during closing and lifting."""
    return all(contacts.get(phase, {}).get(finger, False)
               for phase in ("closing", "lifting")
               for finger in ("gripper_link1", "gripper_link2"))


def main():
    if os.environ.get("ROS_DOMAIN_ID") != "183":
        raise SystemExit("Requires isolated ROS_DOMAIN_ID=183")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lift-scaling",
        type=float,
        default=0.1,
        help="Simulation lift speed and acceleration scaling",
    )
    parser.add_argument("--approach-scaling", type=float, default=0.1)
    parser.add_argument("--grip-seconds", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument(
        "--candidate-id",
        type=int,
        choices=range(16),
        help="Try only this simulated grasp orientation",
    )
    parser.add_argument(
        "--reference-z",
        type=float,
        default=-0.13,
        help="Simulated grip reference height in base frame",
    )
    parser.add_argument("--out", type=Path, default=Path("records/gazebo_grasp/result.json"))
    args = parser.parse_args()
    if not 0.0 < args.lift_scaling <= 0.1 or not 0.0 < args.approach_scaling <= 0.1:
        parser.error("scaling must be in (0, 0.1]")
    if not 2 <= args.grip_seconds <= 20 or not 0 <= args.hold_seconds <= 20:
        parser.error("invalid grip or hold duration")
    if not -0.145 <= args.reference_z <= -0.08:
        parser.error("reference z is outside the benchmark range")
    rclpy.init()
    n = CylinderPreview()
    n.create_subscription(JointState, "/joint_states", lambda m: setattr(n, "joints", m), 10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "finger_contacts": {},
        "backend": "gazebo_contact",
        "lift_scaling": args.lift_scaling,
        "candidate_requested": args.candidate_id,
        "reference_z_m": args.reference_z,
        "pose_source": "known simulator benchmark pose, not vision",
        "attachment_used": False,
        "status_scope": "experimental known-pose contact benchmark, not vision grasp or real-robot validation",
        "stages": [],
    }
    # This sidecar describes the last generated scene, not a query of live Gazebo.
    scene_metadata = Path("/tmp/piper_contact_grasp/scene_parameters.json")
    if scene_metadata.exists():
        report["generated_scene_parameters"] = json.loads(scene_metadata.read_text())
        report["scene_parameters_scope"] = (
            "last generated scene; not independently verified against the running simulator"
        )
    phase = ["initial"]
    last_contact_sample = {}

    def contact(msg, finger):
        """Sample representative contact states; this is not total finger force."""
        hits = [
            s
            for s in msg.states
            if "cylinder::" in s.collision1_name or "cylinder::" in s.collision2_name
        ]
        if hits:
            report["finger_contacts"].setdefault(phase[0], {})[finger] = True
            # Record actual wrench and contact geometry, not merely a boolean contact.
            # Limit output volume while retaining the time course of closing and lifting.
            now = time.monotonic()
            key = (phase[0], finger)
            if now - last_contact_sample.get(key, 0.0) >= 0.1:
                last_contact_sample[key] = now

                def vector(v):
                    return [float(v.x), float(v.y), float(v.z)]

                entries = (
                    report.setdefault("contact_samples", {})
                    .setdefault(phase[0], {})
                    .setdefault(finger, [])
                )
                if len(entries) < 100:
                    strongest = max(
                        hits, key=lambda s: np.linalg.norm(vector(s.total_wrench.force))
                    )
                    entries.append(
                        {
                            "time_monotonic_s": now,
                            "contact_states": len(hits),
                            "points": sum(len(s.contact_positions) for s in hits),
                            "representative_force": vector(strongest.total_wrench.force),
                            "representative_normal": (
                                vector(strongest.contact_normals[0])
                                if strongest.contact_normals
                                else None
                            ),
                            "representative_position": (
                                vector(strongest.contact_positions[0])
                                if strongest.contact_positions
                                else None
                            ),
                            "max_depth": max((v for s in hits for v in s.depths), default=0.0),
                        }
                    )

    for finger in ["gripper_link1", "gripper_link2"]:
        n.create_subscription(
            ContactsState, "/" + finger + "/contacts", lambda m, f=finger: contact(m, f), 10
        )
    entity_client = n.create_client(GetEntityState, "/gazebo/get_entity_state")
    object_pending = [False]

    def sample_object():
        """Record object and finger positions during closing and lifting."""
        if (
            phase[0] not in ("closing", "lifting")
            or object_pending[0]
            or not entity_client.service_is_ready()
        ):
            return
        sample_phase = phase[0]
        object_pending[0] = True
        future = entity_client.call_async(
            GetEntityState.Request(name="cylinder", reference_frame="world")
        )

        def done(f):
            object_pending[0] = False
            result = f.result()
            if result is None or not result.success:
                return
            entries = report.setdefault("object_samples", {}).setdefault(sample_phase, [])
            if len(entries) >= 200:
                return
            pos = result.state.pose.position
            orientation = result.state.pose.orientation
            velocity = result.state.twist.linear
            joints = dict(zip(n.joints.name, n.joints.position)) if n.joints is not None else {}
            entries.append(
                {
                    "time_monotonic_s": time.monotonic(),
                    "xyz": [pos.x, pos.y, pos.z],
                    "orientation_xyzw": [orientation.x, orientation.y, orientation.z, orientation.w],
                    "linear_velocity_xyz": [velocity.x, velocity.y, velocity.z],
                    "finger_joints": {
                        k: joints[k] for k in ("gripper_joint1", "gripper_joint2") if k in joints
                    },
                }
            )

        future.add_done_callback(done)

    n.create_timer(0.1, sample_object)

    def state():
        n.wait(lambda: n.joints is not None, 5)
        for _ in range(5):
            rclpy.spin_once(n, timeout_sec=0.01)
        return RobotState(joint_state=deepcopy(n.joints), is_diff=False)

    def entity():
        r = n.service(
            GetEntityState,
            "/gazebo/get_entity_state",
            GetEntityState.Request(name="cylinder", reference_frame="world"),
        )
        if not r.success:
            raise RuntimeError("No simulated object state")
        return [r.state.pose.position.x, r.state.pose.position.y, r.state.pose.position.z]

    def execute(path, controller):
        """Execute a trajectory and check action completion and joint feedback."""
        print(
            "trajectory",
            controller,
            list(path.joint_names),
            "first",
            list(path.points[0].positions),
            "last",
            list(path.points[-1].positions),
            flush=True,
        )
        client = ActionClient(n, FollowJointTrajectory, f"/{controller}/follow_joint_trajectory")
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError("No simulated controller")
        f = client.send_goal_async(FollowJointTrajectory.Goal(trajectory=path))
        n.wait(f.done, 5)
        goal = f.result()
        if not goal.accepted:
            raise RuntimeError("Trajectory rejected")
        f = goal.get_result_async()
        try:
            n.wait(f.done, 35)
        except Exception:
            cancel = goal.cancel_goal_async()
            n.wait(cancel.done, 3)
            raise
        r = f.result()
        if r.status != 4 or r.result.error_code != 0:
            raise RuntimeError(f"Trajectory failed: {r.status} {r.result.error_string}")
        client.destroy()
        wanted = dict(zip(path.joint_names, path.points[-1].positions))

        def at_goal():
            actual = dict(zip(n.joints.name, n.joints.position)) if n.joints is not None else {}
            # The virtual gripper target can arrive while contact blocks the fingers.
            # Actual finger positions are recorded below; this is not proof of a grasp.
            required = set(wanted) if controller == "arm_controller" else {"gripper"}
            return required.issubset(actual) and all(
                abs(actual[name] - wanted[name]) < 0.015 for name in required
            )

        n.wait(at_goal, 3)
        report.setdefault("executed_end_states", []).append(
            dict(zip(n.joints.name, n.joints.position))
        )

    def command_gripper(opening_m):
        """Command the virtual width and two simulated prismatic finger joints.

        The object may stop the physical fingers before they reach these targets.
        This is position control, not a request for a particular force.
        """
        point = JointTrajectoryPoint(positions=[opening_m, opening_m / 2, -opening_m / 2])
        point.time_from_start.sec = args.grip_seconds
        trajectory = JointTrajectory(joint_names=list(GRIPPER_JOINTS), points=[point])
        execute(trajectory, "gripper_controller")

    def preplan(matrix):
        initial = state()
        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm"
        req.ik_request.ik_link_name = "tcp_link"
        req.ik_request.robot_state = initial
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.pose_stamped.pose = pose_from_matrix(matrix)
        req.ik_request.timeout.sec = 1
        ik = n.service(GetPositionIK, "/compute_ik", req)
        if ik.error_code.val != 1:
            raise RuntimeError(f"Pregrasp IK {ik.error_code.val}")
        req = GetMotionPlan.Request()
        m = req.motion_plan_request
        m.group_name = "arm"
        m.start_state = initial
        m.allowed_planning_time = 3.0
        m.max_velocity_scaling_factor = 0.1
        m.max_acceleration_scaling_factor = 0.1
        c = Constraints()
        for name, value in zip(ik.solution.joint_state.name, ik.solution.joint_state.position):
            if name.startswith("joint"):
                c.joint_constraints.append(
                    JointConstraint(
                        joint_name=name,
                        position=value,
                        tolerance_above=0.001,
                        tolerance_below=0.001,
                        weight=1.0,
                    )
                )
        m.goal_constraints = [c]
        r = n.service(GetMotionPlan, "/plan_kinematic_path", req).motion_plan_response
        if r.error_code.val != 1:
            raise RuntimeError(f"Pregrasp plan {r.error_code.val}")
        return r.trajectory.joint_trajectory

    def cartesian(matrix, scaling=0.1):
        q = GetCartesianPath.Request()
        q.header.frame_id = "base_link"
        q.group_name = "arm"
        q.link_name = "tcp_link"
        q.start_state = state()
        q.waypoints = [pose_from_matrix(matrix)]
        q.max_step = 0.003
        q.avoid_collisions = True
        q.max_velocity_scaling_factor = scaling
        q.max_acceleration_scaling_factor = scaling
        r = n.service(GetCartesianPath, "/compute_cartesian_path", q)
        if r.error_code.val != 1 or r.fraction < 0.999:
            raise RuntimeError(f"Cartesian fraction {r.fraction}, code {r.error_code.val}")
        return r.solution.joint_trajectory

    try:
        n.wait(lambda: n.count_publishers("/joint_states") == 1, 10)
        if n.count_publishers("/feedback/joint_states"):
            raise RuntimeError("Real feedback detected; refuse")
        move_group = n.create_client(SetParameters, "/move_group/set_parameters")
        if not move_group.wait_for_service(timeout_sec=15):
            raise RuntimeError("MoveIt parameter service unavailable after startup")
        n.destroy_client(move_group)
        n.service(
            SetParameters,
            "/move_group/set_parameters",
            SetParameters.Request(
                parameters=[
                    Parameter(
                        name="use_sim_time",
                        value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=True),
                    )
                ]
            ),
        )
        report["initial_object_xyz"] = entity()
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.robot_state.is_diff = True
        shape = report.get("generated_scene_parameters", {}).get("object_shape", "cylinder")
        target_type = SolidPrimitive.BOX if shape == "cube" else SolidPrimitive.CYLINDER
        target_dims = [0.046, 0.046, 0.046] if shape == "cube" else [0.08, 0.023]
        for name, typ, dims, xyz in [
            ("ground", SolidPrimitive.BOX, [2.0, 2.0, 0.05], [0.0, 0.0, -0.175]),
            ("target", target_type, target_dims, report["initial_object_xyz"]),
        ]:
            o = CollisionObject(id=name, operation=CollisionObject.ADD)
            o.header.frame_id = "base_link"
            o.primitives = [SolidPrimitive(type=typ, dimensions=dims)]
            t = np.eye(4)
            t[:3, 3] = xyz
            o.primitive_poses = [pose_from_matrix(t)]
            req.scene.world.collision_objects.append(o)
        # Permit only finger/object contact; all other collision pairs stay checked.
        semantic = (
            n.service(
                GetParameters,
                "/move_group/get_parameters",
                GetParameters.Request(names=["robot_description_semantic"]),
            )
            .values[0]
            .string_value
        )
        model = ET.fromstring(semantic)
        allowed = {
            frozenset((p.get("link1"), p.get("link2"))) for p in model.findall("disable_collisions")
        }
        allowed.update(
            frozenset(pair)
            for pair in [
                ("target", "gripper_link1"),
                ("target", "gripper_link2"),
                ("ground", "base_link"),
            ]
        )
        names = sorted(set().union(*allowed))
        req.scene.allowed_collision_matrix.entry_names = names
        for a in names:
            req.scene.allowed_collision_matrix.entry_values.append(
                AllowedCollisionEntry(
                    enabled=[a == b or frozenset((a, b)) in allowed for b in names]
                )
            )
        if not n.service(ApplyPlanningScene, "/apply_planning_scene", req).success:
            raise RuntimeError("Scene rejected")
        command_gripper(OPEN_WIDTH_M)
        report["stages"].append("opened")
        # Model tip is 142.5 mm beyond link6. tcp_link is flange in this model.
        candidates = grasp_candidates(
            {
                "axis_reference_xyz_m": [0.45, 0.0, args.reference_z],
                "axis_direction": [0.0, 0.0, 1.0],
                "diameter_m": 0.046,
                "observed_red_span_m": 0.06,
            },
            model_tip_transform(installed_gripper_model_path()),
            np.eye(4),
        )
        if args.candidate_id is not None:
            candidates = [c for c in candidates if c["id"] == args.candidate_id]
        path = None
        report["candidate_attempts"] = []
        for candidate in candidates:
            try:
                path = preplan(np.array(candidate["T_base_pregrasp_tcp"]))
                target = np.array(candidate["T_base_tcp"])
                report["selected_candidate"] = candidate
                break
            except RuntimeError as exc:
                report["candidate_attempts"].append({"id": candidate["id"], "reason": str(exc)})
        if path is None:
            raise RuntimeError("No reachable pregrasp candidate")
        execute(path, "arm_controller")
        report["stages"].append("pregrasp")
        execute(cartesian(target, args.approach_scaling), "arm_controller")
        report["stages"].append("approached")
        print("approached", flush=True)
        phase[0] = "closing"
        command_gripper(CLOSED_WIDTH_M)
        report["stages"].append("closed")
        print("closed", flush=True)
        report["after_close_xyz"] = entity()
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(n, timeout_sec=0.05)
        phase[0] = "lifting"
        lift = target.copy()
        lift[2, 3] += 0.10
        execute(cartesian(lift, args.lift_scaling), "arm_controller")
        report["stages"].append("lifted_arm")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            rclpy.spin_once(n, timeout_sec=0.05)
        report["final_object_xyz"] = entity()
        report["object_lift_m"] = report["final_object_xyz"][2] - report["initial_object_xyz"][2]
        report["success"] = (
            both_fingers_in_both_phases(report["finger_contacts"])
            and report["object_lift_m"] > 0.07
            and bool(np.linalg.norm(np.array(report["final_object_xyz"][:2]) - [0.45, 0]) < 0.06)
        )
        # Cylinder acceptance also requires it to remain upright and stable.
        # Height alone can pass even when the cylinder rotates onto its side.
        if shape == "cylinder":
            tail = report.get("object_samples", {}).get("lifting", [])[-10:]
            if len(tail) >= 10:
                def tilt_degrees(sample):
                    qx, qy, _, _ = sample["orientation_xyzw"]
                    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx*qx + qy*qy)))))
                angles = [tilt_degrees(sample) for sample in tail]
                origin = np.array(tail[0]["xyz"])
                motion = max(float(np.linalg.norm(np.array(sample["xyz"]) - origin)) for sample in tail)
                report["cylinder_hold_check"] = {
                    "samples": len(tail),
                    "max_tilt_deg": max(angles),
                    "max_displacement_m": motion,
                    "limits": {"tilt_deg": 15, "displacement_m": 0.003},
                }
                report["success"] = bool(report["success"] and max(angles) < 15 and motion < 0.003)
            else:
                report["cylinder_hold_check"] = {"samples": len(tail), "error": "insufficient hold samples"}
                report["success"] = False
        report["status"] = "completed"
    except Exception as e:
        report.update(status="failed", success=False, reason=str(e))
        print(str(e), flush=True)
    finally:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        n.destroy_node()
        rclpy.shutdown()
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
