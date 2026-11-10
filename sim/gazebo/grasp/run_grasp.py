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
from rclpy.parameter import Parameter as RclParameter
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import (
    AttachedCollisionObject,
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
from piperx_lab.collision_geometry import collision_object_from_hull
from piperx_lab.cylinder_preview import CylinderPreview, pose_from_matrix
from piperx_lab.cylinder_grasp import grasp_candidates, model_tip_transform
from piperx_lab.grasp_candidate_filter import (
    candidate_center_near_target,
    neighborhood_radius_m,
    validate_candidates_payload,
)
from piperx_lab.grasp_candidates import (
    MAX_OPENING_M,
    apply_revalidation,
    geometric_candidates,
)
from piperx_lab.gripper_policy import MIN_CLOSE_WIDTH_M, classify_grasp, closing_sequence
from piperx_lab.model import installed_gripper_model_path
from piperx_lab.object_observation import (
    SOURCE as GENERIC_OBSERVATION_SOURCE,
    GenericObjectObservation,
    read_ply,
)

# These are target widths, not measured contact forces.
OPEN_WIDTH_M = 0.070
GRIPPER_JOINTS = ("gripper", "gripper_joint1", "gripper_joint2")
DEFAULT_NEIGHBORHOOD_RADIUS_M = 0.09


def both_fingers_in_both_phases(contacts):
    """Require evidence from each finger during closing and lifting."""
    return all(contacts.get(phase, {}).get(finger, False)
               for phase in ("closing", "lifting")
               for finger in ("gripper_link1", "gripper_link2"))


def planning_center_from_observation(observation):
    """Validate the vision-only planner input without consulting simulator truth."""
    if observation.get("status") != "ok" or observation.get("base_frame") != "base_link":
        raise ValueError("Expected a valid simulated RGB-D target observation")
    if observation.get("source") == GENERIC_OBSERVATION_SOURCE:
        parsed = GenericObjectObservation.from_dict(observation)
        return parsed.center()
    if observation.get("source") not in {"sim_rgbd_red_cylinder_fit", "sim_rgbd_red_cube_fit"}:
        raise ValueError("Expected a valid simulated RGB-D target observation")
    center = np.asarray(observation.get("center_xyz_m"), dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Invalid observed target center")
    return center.tolist()


def observation_dimensions(observation):
    """Observed object dimensions in metres, or None when unavailable."""
    if observation is None:
        return None
    dims = observation.get("dimensions_m")
    if isinstance(dims, list) and len(dims) == 3:
        try:
            values = [float(v) for v in dims]
        except (TypeError, ValueError):
            return None
        if all(math.isfinite(v) and v > 0 for v in values):
            return values
    return None


def load_observation_cloud(args):
    """Base-frame object cloud referenced by a generic observation, or None."""
    if args.vision_observation is None:
        return None
    observation = json.loads(args.vision_observation.read_text())
    relative = observation.get("points_base_path")
    if not relative:
        return None
    with np.load(args.vision_observation.parent / relative, allow_pickle=False) as npz:
        points = np.asarray(npz["points_base"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Invalid base-frame object cloud")
    return points


def hold_mode_from_observation(observation, legacy_shape):
    """Which hold gating applies: upright tilt, displacement only, or none."""
    if observation is not None and observation.get("source") == GENERIC_OBSERVATION_SOURCE:
        return GenericObjectObservation.from_dict(observation).hold_check_mode()
    return "upright_tilt" if legacy_shape == "cylinder" else "none"


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
    parser.add_argument("--vision-observation", type=Path, help="Use only a fresh simulated RGB-D observation for planning")
    parser.add_argument("--graspnet-candidates", type=Path,
                        help="Camera-derived candidate TCP poses; simulation only; requires --vision-observation")
    parser.add_argument("--geometry-fallback", action="store_true",
                        help="Append geometric candidates after the network candidates all fail")
    parser.add_argument("--geometry-candidates", action="store_true",
                        help="Fusion mode: also try geometric antipodal candidates alongside network candidates")
    args = parser.parse_args()
    if args.graspnet_candidates and not args.vision_observation:
        parser.error("--graspnet-candidates requires --vision-observation")
    if args.graspnet_candidates and args.candidate_id is not None:
        parser.error("--candidate-id applies only to the geometric candidate generator")
    if args.geometry_fallback and (args.vision_observation is None or args.graspnet_candidates is None):
        parser.error("--geometry-fallback requires --vision-observation and --graspnet-candidates")
    if not 0.0 < args.lift_scaling <= 0.1 or not 0.0 < args.approach_scaling <= 0.1:
        parser.error("scaling must be in (0, 0.1]")
    if not 2 <= args.grip_seconds <= 20 or not 0 <= args.hold_seconds <= 20:
        parser.error("invalid grip or hold duration")
    if not -0.145 <= args.reference_z <= -0.08:
        parser.error("reference z is outside the benchmark range")
    rclpy.init()
    n = CylinderPreview()
    n.set_parameters([RclParameter("use_sim_time", RclParameter.Type.BOOL, True)])
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
        "physical_grasp_success": False,
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
    observation = None
    generic_observation = None
    table_height = None
    graspnet_data = None
    graspnet_candidates = None
    if args.vision_observation is not None:
        observation = json.loads(args.vision_observation.read_text())
        planning_center_from_observation(observation)
        if observation.get("source") == GENERIC_OBSERVATION_SOURCE:
            generic_observation = GenericObjectObservation.from_dict(observation)
            report["pose_source"] = ("simulated RGB-D; shape-agnostic observation with "
                                     "convex-hull collision geometry")
        else:
            report["pose_source"] = "simulated RGB-D; known upright dimensions and table height"
        report["status_scope"] = "vision-based simulation benchmark, not a real-robot validation"
        report["observation"] = observation
    if args.graspnet_candidates is not None:
        graspnet_data = json.loads(args.graspnet_candidates.read_text())
        graspnet_candidates = validate_candidates_payload(graspnet_data)
        report["pose_source"] = ("simulated RGB-D GraspNet candidates; observed convex-hull "
                                 "collision geometry"
                                 if generic_observation is not None else
                                 "simulated RGB-D GraspNet candidates; observed cylinder collision geometry")
        report["graspnet_input"] = str(args.graspnet_candidates)
        report["graspnet_candidate_count"] = len(graspnet_data["candidates"])
        report["graspnet_width_compatible_count"] = len(graspnet_candidates)
        dims = observation_dimensions(observation)
        radius = neighborhood_radius_m(dims) if dims is not None else DEFAULT_NEIGHBORHOOD_RADIUS_M
        center = planning_center_from_observation(observation) if observation is not None else None
        cloud = load_observation_cloud(args) if generic_observation is not None else None
        hull_vertices = None
        if cloud is not None and generic_observation.convex_hull_path:
            hull_vertices, _ = read_ply(
                args.vision_observation.parent / generic_observation.convex_hull_path)
        if cloud is not None:
            # Single validation path for every candidate, network or rotated:
            # width feasibility, jaw containment, bilateral contact support.
            annotated = apply_revalidation(graspnet_candidates, cloud,
                                           hull_vertices=hull_vertices,
                                           table_height_m=table_height)
            graspnet_candidates = [c for c in annotated if c.get("rejection_reason") is None]
            report["revalidation_rejections"] = {
                str(reason): sum(1 for c in annotated
                                 if c.get("rejection_reason") == reason)
                for reason in {c.get("rejection_reason") for c in annotated
                               if c.get("rejection_reason")}}
            report["revalidated_candidate_count"] = len(graspnet_candidates)
        if center is not None:
            kept = [c for c in graspnet_candidates
                    if candidate_center_near_target(c, center, radius)]
            report["neighborhood_filtered_count"] = len(graspnet_candidates) - len(kept)
            report["neighborhood_radius_m"] = radius
            if not kept:
                raise ValueError("No candidate grip centre lies near the observed target")
            graspnet_candidates = kept
        if cloud is not None:
            graspnet_candidates.sort(key=lambda c: (-float(c.get("confidence") or 0.0),
                                                    float(c.get("approach_distance_m") or 1.0)))
        report["planning_limitations"] = [
            "finger contact with the target is permitted only during the final approach segment",
            "prevalidated Cartesian trajectories are replanned from feedback before execution",
            "joint consistency does not prove the target stayed stationary between observations",
            "the lift is planned with the payload attached for prediction; Gazebo still decides "
            "the grasp through real contact and friction",
        ]
        report["planning_limitations"] = [
            "finger contact with the target is permitted only during the final approach segment",
            "preplanned lift does not include the object attached to the gripper",
            "prevalidated Cartesian trajectories are replanned from feedback before execution",
            "joint consistency does not prove the target stayed stationary between observations",
        ]
    phase = ["initial"]
    main_stage = ["startup"]
    stage_timings = {}
    t_zero = time.monotonic()

    def mark(name):
        """Cumulative seconds since startup for the stage report."""
        stage_timings[name] = round(time.monotonic() - t_zero, 3)
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
        # is_diff=True: the state overlays the current planning scene, so an
        # attached payload survives into IK/planning/cartesian start states.
        # A full (is_diff=False) state would silently drop the attachment.
        return RobotState(joint_state=deepcopy(n.joints), is_diff=True)

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

        settled = True
        try:
            n.wait(at_goal, 3)
        except TimeoutError:
            settled = False
            actual = dict(zip(n.joints.name, n.joints.position)) if n.joints is not None else {}
            required = set(wanted) if controller == "arm_controller" else {"gripper"}
            report["feedback_failure"] = {
                "controller": controller,
                "required_joint_error_rad_or_m": {
                    name: float(actual[name] - wanted[name]) if name in actual else None
                    for name in sorted(required)
                },
                "tolerance": 0.015,
            }
        report.setdefault("executed_end_states", []).append(
            dict(zip(n.joints.name, n.joints.position))
        )
        return settled

    def command_gripper(opening_m, seconds=None):
        """Command the virtual width and two simulated prismatic finger joints.

        The object may stop the physical fingers before they reach these targets.
        This is position control, not a request for a particular force.
        """
        point = JointTrajectoryPoint(positions=[opening_m, opening_m / 2, -opening_m / 2])
        point.time_from_start.sec = seconds or args.grip_seconds
        trajectory = JointTrajectory(joint_names=list(GRIPPER_JOINTS), points=[point])
        execute(trajectory, "gripper_controller")

    def preplan(matrix, initial=None):
        initial = state() if initial is None else initial
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

    def cartesian(matrix, scaling=0.1, initial=None):
        q = GetCartesianPath.Request()
        q.header.frame_id = "base_link"
        q.group_name = "arm"
        q.link_name = "tcp_link"
        q.start_state = state() if initial is None else initial
        q.waypoints = [pose_from_matrix(matrix)]
        q.max_step = 0.003
        q.avoid_collisions = True
        q.max_velocity_scaling_factor = scaling
        q.max_acceleration_scaling_factor = scaling
        r = n.service(GetCartesianPath, "/compute_cartesian_path", q)
        if r.error_code.val != 1 or r.fraction < 0.999:
            raise RuntimeError(f"Cartesian fraction {r.fraction}, code {r.error_code.val}")
        return r.solution.joint_trajectory

    def state_at_end(initial, trajectory):
        """Overlay arm trajectory endpoint on complete state (including fingers)."""
        if not trajectory.points or len(trajectory.joint_names) != len(trajectory.points[-1].positions):
            raise RuntimeError("Empty or invalid trajectory endpoint")
        result = deepcopy(initial)
        locations = {name: i for i, name in enumerate(result.joint_state.name)}
        positions = list(result.joint_state.position)
        for name, value in zip(trajectory.joint_names, trajectory.points[-1].positions):
            if name not in locations:
                raise RuntimeError(f"Trajectory joint {name} missing from state")
            positions[locations[name]] = value
        result.joint_state.position = positions
        return result

    try:
        n.wait(lambda: n.count_publishers("/joint_states") == 1 and n.joints is not None
               and (observation is None or n.get_clock().now().nanoseconds > 0), 10)
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
        if observation is not None:
            sample_stamp = observation["stamp"]["sec"] + observation["stamp"]["nanosec"] * 1e-9
            age = n.get_clock().now().nanoseconds * 1e-9 - sample_stamp
            if not 0 <= age <= 5:
                raise ValueError(f"Vision observation is stale ({age:.2f} simulated seconds)")
            current = dict(zip(n.joints.name, n.joints.position))
            saved = observation["joint_positions_rad"]
            if any(name not in saved or name not in current or abs(saved[name] - current[name]) > .02
                   for name in (f"joint{i}" for i in range(1, 7))):
                raise ValueError("Arm state changed since the vision observation")
            if graspnet_data is not None:
                stamp = graspnet_data.get("sample_stamp", {})
                model_stamp = float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
                model_age = n.get_clock().now().nanoseconds * 1e-9 - model_stamp
                model_joints = graspnet_data.get("sample_joint_positions_rad", {})
                if not math.isfinite(model_stamp) or not 0 <= model_age <= 5:
                    raise ValueError(f"GraspNet RGB-D sample is stale ({model_age:.2f} simulated seconds)")
                if abs(model_stamp - sample_stamp) > 3.0:
                    raise ValueError("GraspNet sample and cylinder observation differ by over 3 seconds")
                if any(name not in model_joints or name not in current
                       or not math.isfinite(float(model_joints[name]))
                       or abs(float(model_joints[name]) - current[name]) > .02
                       for name in (f"joint{i}" for i in range(1, 7))):
                    raise ValueError("Arm state changed since the GraspNet RGB-D sample")
                report["graspnet_sample_age_sim_s"] = model_age
                report["graspnet_observation_gap_sim_s"] = abs(model_stamp - sample_stamp)
        # Gazebo object state is recorded only for benchmark evaluation, never for target planning.
        report["initial_object_xyz"] = entity()
        if observation is not None:
            estimated_center = planning_center_from_observation(observation)
            report["vision_error_m"] = float(np.linalg.norm(np.array(estimated_center) - report["initial_object_xyz"]))
        else:
            estimated_center = list(report["initial_object_xyz"])
        main_stage[0] = "planning_scene"
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.robot_state.is_diff = True
        legacy_shape = (observation.get("shape") if observation is not None else None) or \
            report.get("generated_scene_parameters", {}).get("object_shape", "cylinder")
        target_o = CollisionObject(id="target", operation=CollisionObject.ADD)
        target_o.header.frame_id = "base_link"
        if generic_observation is not None:
            # Collision geometry comes from the observed hull; low-quality
            # geometry is rejected, never replaced by assumed sizes.
            shape = "mesh"
            if not generic_observation.convex_hull_path:
                raise ValueError("Generic observation carries no convex hull path")
            hull_path = (args.vision_observation.parent / generic_observation.convex_hull_path)
            table_height = float(generic_observation.quality["table_height_m"])
            target_mesh, provenance = collision_object_from_hull(
                hull_path, table_height_m=table_height,
                observation=generic_observation.to_dict())
            target_o.meshes = target_mesh.meshes
            target_o.mesh_poses = target_mesh.mesh_poses
            report["collision_geometry"] = provenance
        else:
            shape = legacy_shape
            target_type = SolidPrimitive.BOX if shape == "cube" else SolidPrimitive.CYLINDER
            if observation is not None and observation.get("dimensions_m") is not None:
                dimensions = [float(value) for value in observation["dimensions_m"]]
                target_dims = dimensions if shape == "cube" else [dimensions[2], dimensions[0] / 2]
            else:
                target_dims = [0.046, 0.046, 0.046] if shape == "cube" else [0.08, 0.023]
            target_o.primitives = [SolidPrimitive(type=target_type, dimensions=target_dims)]
            t = np.eye(4)
            t[:3, 3] = estimated_center
            target_o.primitive_poses = [pose_from_matrix(t)]
        for name, obj in [
            ("ground", None),
            ("target", target_o),
        ]:
            if obj is None:
                obj = CollisionObject(id=name, operation=CollisionObject.ADD)
                obj.header.frame_id = "base_link"
                obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[2.0, 2.0, 0.05])]
                t = np.eye(4)
                t[:3, 3] = [0.0, 0.0, -0.175]
                obj.primitive_poses = [pose_from_matrix(t)]
            req.scene.world.collision_objects.append(obj)
        # Observed non-target clusters (blockers, other objects) become
        # collision boxes; this is perception input, never scene truth.
        for obstacle_index, box in enumerate((observation or {}).get("obstacles_base") or []):
            low = np.asarray(box["aabb_min_m"], dtype=float)
            high = np.asarray(box["aabb_max_m"], dtype=float)
            if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
                continue
            obstacle = CollisionObject(id=f"obstacle_{obstacle_index}",
                                       operation=CollisionObject.ADD)
            obstacle.header.frame_id = "base_link"
            obstacle.primitives = [SolidPrimitive(
                type=SolidPrimitive.BOX, dimensions=np.maximum(high - low, 0.005).tolist())]
            t = np.eye(4)
            t[:3, 3] = (low + high) / 2
            obstacle.primitive_poses = [pose_from_matrix(t)]
            req.scene.world.collision_objects.append(obstacle)
        report["planning_obstacles"] = len((observation or {}).get("obstacles_base") or [])
        report["object_dimensions_m"] = observation_dimensions(observation)
        report["object_shape"] = shape
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

        def allow_target_finger_contact(enabled):
            """Change only fingertip/target permissions in the known scene matrix."""
            diff = ApplyPlanningScene.Request()
            diff.scene.is_diff = True
            diff.scene.robot_state.is_diff = True
            diff.scene.allowed_collision_matrix.entry_names = names
            for a in names:
                row = []
                for b in names:
                    pair = frozenset((a, b))
                    is_target_finger = pair in {
                        frozenset(("target", "gripper_link1")),
                        frozenset(("target", "gripper_link2")),
                    }
                    row.append(enabled if is_target_finger else a == b or pair in allowed)
                diff.scene.allowed_collision_matrix.entry_values.append(AllowedCollisionEntry(enabled=row))
            if not n.service(ApplyPlanningScene, "/apply_planning_scene", diff).success:
                raise RuntimeError("Target contact permission update rejected")

        command_gripper(OPEN_WIDTH_M)
        report["stages"].append("opened")
        mark("opened")
        observation_cloud = load_observation_cloud(args) if generic_observation is not None else None
        if generic_observation is not None:
            table_height = float(generic_observation.quality["table_height_m"])

        def geometry_candidate_set(tag_source):
            """Antipodal hypotheses from observed geometry only; shared validation."""
            if observation_cloud is None or table_height is None:
                return []
            tip_model = model_tip_transform(installed_gripper_model_path())
            tcp_from_tip = np.linalg.inv(tip_model)  # flange relative to fingertip midpoint
            generated = geometric_candidates(generic_observation.to_dict(), observation_cloud,
                                             table_height_m=table_height,
                                             hull_vertices=hull_vertices,
                                             T_tip_tcp=tcp_from_tip)
            if generated:
                generated = apply_revalidation(generated, observation_cloud,
                                               hull_vertices=hull_vertices,
                                               table_height_m=table_height)
            for candidate in generated:
                candidate["source"] = tag_source
                candidate["method"] = "geometry"
                if tag_source == "geometric_symmetry_fallback":
                    candidate["fallback"] = True
                    candidate["kind"] = "geometric_symmetry_fallback"
            return generated

        # Model tip is 142.5 mm beyond link6. tcp_link is flange in this model.
        if graspnet_candidates is not None:
            candidates = list(graspnet_candidates)
            if args.geometry_candidates:
                fusion = geometry_candidate_set("geometric_antipodal")
                report["geometry_fusion_candidate_count"] = len(fusion)
                candidates = sorted(candidates + fusion,
                                    key=lambda c: (-float(c.get("confidence") or 0.0),
                                                   float(c.get("approach_distance_m") or 1.0)))
        else:
            # Known-pose benchmark mode (no vision candidates supplied).
            candidates = grasp_candidates(
                {
                    "axis_reference_xyz_m": ([estimated_center[0], estimated_center[1], estimated_center[2]]
                                             if observation is not None else [0.45, 0.0, args.reference_z]),
                    "axis_direction": [0.0, 0.0, 1.0],
                    "diameter_m": 0.046,
                    "observed_red_span_m": 0.06,
                },
                model_tip_transform(installed_gripper_model_path()),
                np.eye(4),
            )
        if args.candidate_id is not None:
            candidates = [c for c in candidates if c["id"] == args.candidate_id]
        if args.geometry_fallback:
            fallback = geometry_candidate_set("geometric_symmetry_fallback")
            if not fallback and generic_observation is None:
                # Legacy fixed-shape regression fallback (no cloud available).
                reference = list(estimated_center)
                if shape == "cube":
                    reference[2] -= 0.013
                fallback = grasp_candidates(
                    {
                        "axis_reference_xyz_m": reference,
                        "axis_direction": [0.0, 0.0, 1.0],
                        "diameter_m": 0.046,
                        "observed_red_span_m": 0.06,
                    },
                    model_tip_transform(installed_gripper_model_path()),
                    np.eye(4),
                )
                for candidate in fallback:
                    candidate["id"] += 100
                    candidate["kind"] = "geometric_symmetry_fallback"
                    candidate["fallback"] = True
            report["geometry_fallback_candidate_count"] = len(fallback)
            candidates.extend(fallback)
        candidates = candidates[:40]
        main_stage[0] = "candidate_planning"
        mark("planning_scene")
        path = None
        report["candidate_attempts"] = []
        for candidate in candidates:
            stage = "pregrasp"
            try:
                if graspnet_candidates is not None:
                    allow_target_finger_contact(False)
                initial = state()
                candidate_path = preplan(np.array(candidate["T_base_pregrasp_tcp"]), initial)
                target = np.array(candidate["T_base_tcp"])
                if graspnet_candidates is not None:
                    # Keep target/finger collisions disabled until the final 10 mm
                    # of approach, where intended contact is allowed.
                    pre_end = state_at_end(initial, candidate_path)
                    start_xyz = np.array(candidate["T_base_pregrasp_tcp"])[:3, 3]
                    delta = start_xyz - target[:3, 3]
                    length = np.linalg.norm(delta)
                    near = target.copy()
                    near[:3, 3] += delta / length * min(.01, length / 4)
                    stage = "approach_without_target_contact"
                    approach_path = cartesian(near, args.approach_scaling, pre_end)
                    near_end = state_at_end(pre_end, approach_path)
                    stage = "final_approach_with_finger_contact"
                    allow_target_finger_contact(True)
                    final_path = cartesian(target, args.approach_scaling, near_end)
                    approach_end = state_at_end(near_end, final_path)
                    lift_target = target.copy()
                    lift_target[2, 3] += .10
                    stage = "predicted_lift_with_payload"
                    cartesian(lift_target, args.lift_scaling, approach_end)
                path = candidate_path
                report["selected_candidate"] = candidate
                report["candidate_attempts"].append({"id": candidate["id"],
                    "source": candidate.get("source"), "method": candidate.get("method"),
                    "confidence": candidate.get("confidence"),
                    "stage": "prevalidated" if graspnet_candidates is not None else "pregrasp_planned",
                    "status": "selected"})
                break
            except (RuntimeError, TimeoutError) as exc:
                report["candidate_attempts"].append({"id": candidate["id"],
                    "source": candidate.get("source"), "method": candidate.get("method"),
                    "stage": stage, "status": "rejected", "reason": str(exc)})
                if graspnet_candidates is not None:
                    # Do not leave the scene permissive if the last candidate
                    # fails after entering the intended-contact stage.
                    allow_target_finger_contact(False)
        if path is None:
            raise RuntimeError("No candidate passed pregrasp and required Cartesian checks")
        if graspnet_candidates is not None:
            allow_target_finger_contact(False)
        main_stage[0] = "pregrasp_execution"
        if not execute(path, "arm_controller"):
            raise RuntimeError("Pregrasp trajectory completed without reaching joint tolerance")
        report["stages"].append("pregrasp")
        main_stage[0] = "approach_execution"
        if graspnet_candidates is not None:
            if not execute(cartesian(near, args.approach_scaling), "arm_controller"):
                raise RuntimeError("Approach trajectory completed without reaching joint tolerance")
            report["stages"].append("approached_without_contact")
            allow_target_finger_contact(True)
        if not execute(cartesian(target, args.approach_scaling), "arm_controller"):
            raise RuntimeError("Final approach completed without reaching joint tolerance")
        report["stages"].append("approached")
        print("approached", flush=True)
        main_stage[0] = "closing"
        mark("approached")
        phase[0] = "closing"
        # Progressive closing driven by contact feedback: step the width down
        # from just outside the predicted object width and stop as soon as
        # both fingers register contact, instead of commanding a fixed
        # overshoot that over-presses small parts.
        selected_candidate = report.get("selected_candidate") or {}
        sequence = closing_sequence(selected_candidate.get("predicted_width_m"))
        closing_contacts = report["finger_contacts"].setdefault("closing", {})
        close_width = sequence[-1]
        stopped_at_step = len(sequence) - 1
        for step_index, width in enumerate(sequence):
            command_gripper(width, seconds=4 if step_index == 0 else 2)
            close_width = width
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline:
                rclpy.spin_once(n, timeout_sec=0.05)
            if (closing_contacts.get("gripper_link1")
                    and closing_contacts.get("gripper_link2")):
                # First contact only barely loads the position-controlled
                # fingers. The fingers may have stopped on the object well
                # short of the commanded width, so the locking step retreats
                # from the MEASURED finger width, not from the command;
                # otherwise the extra squeeze can be a fraction of a
                # millimetre and the hold slips on lift.
                stopped_at_step = step_index
                actual_width = None
                if n.joints is not None:
                    joints_now = dict(zip(n.joints.name, n.joints.position))
                    if "gripper_joint1" in joints_now and "gripper_joint2" in joints_now:
                        actual_width = (joints_now["gripper_joint1"]
                                        - joints_now["gripper_joint2"])
                base_width = actual_width if (actual_width is not None
                                              and 0 < actual_width < width) else width
                lock_width = max(base_width - 0.003, MIN_CLOSE_WIDTH_M)
                if lock_width < base_width:
                    command_gripper(lock_width, seconds=2)
                    close_width = lock_width
                break
        report["closing_stopped_at_step"] = stopped_at_step
        report["commanded_close_width_m"] = close_width
        report["closing_sequence_m"] = sequence
        report["closing_stopped_at_step"] = step_index
        report["stages"].append("closed")
        mark("closed")
        print("closed", flush=True)
        report["after_close_xyz"] = entity()
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(n, timeout_sec=0.05)
        phase[0] = "lifting"
        main_stage[0] = "lift_planning"
        # Attach the observed target to the gripper so the lift plan predicts
        # payload collisions. Gazebo still decides the real grasp by contact.
        # This moveit_msgs build carries attachments on RobotState and names
        # the attach operation APPEND.
        attach = ApplyPlanningScene.Request()
        attach.scene.is_diff = True
        attach.scene.robot_state.is_diff = True
        attach.scene.robot_state.attached_collision_objects.append(
            AttachedCollisionObject(
                link_name="gripper_link2",
                touch_links=["gripper_link1", "gripper_link2", "gripper_link",
                             "link6", "tcp_link"],
                object=CollisionObject(id="target", operation=CollisionObject.APPEND),
            )
        )
        if not n.service(ApplyPlanningScene, "/apply_planning_scene", attach).success:
            raise RuntimeError("Payload attachment rejected")
        report["stages"].append("payload_attached")
        main_stage[0] = "lift_execution"
        lift = target.copy()
        lift[2, 3] += 0.10
        report["lift_feedback_settled"] = execute(
            cartesian(lift, args.lift_scaling), "arm_controller")
        report["stages"].append("lifted_arm")
        mark("lifted")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            rclpy.spin_once(n, timeout_sec=0.05)
        report["final_object_xyz"] = entity()
        report["object_lift_m"] = report["final_object_xyz"][2] - report["initial_object_xyz"][2]
        # Physical grasp: two-finger contact in both phases and the object
        # actually lifted and kept. Reported separately from hold stability
        # and from joint-feedback convergence.
        report["physical_grasp_success"] = bool(
            both_fingers_in_both_phases(report["finger_contacts"])
            and report["object_lift_m"] > 0.07
            and bool(np.linalg.norm(np.array(report["final_object_xyz"][:2]) - np.array(report["initial_object_xyz"][:2])) < 0.06)
        )
        report["success"] = report["physical_grasp_success"]
        # Tip-over gating only where it is physically meaningful; the final
        # full pose is recorded either way for post-hoc evaluation.
        hold_mode = hold_mode_from_observation(
            observation, None if generic_observation is not None else shape)
        main_stage[0] = "hold_evaluation"
        report["hold_check_mode"] = hold_mode
        if hold_mode != "none":
            tail = report.get("object_samples", {}).get("lifting", [])[-10:]
            if len(tail) >= 10:
                def tilt_degrees(sample):
                    qx, qy, _, _ = sample["orientation_xyzw"]
                    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx*qx + qy*qy)))))
                angles = [tilt_degrees(sample) for sample in tail]
                origin = np.array(tail[0]["xyz"])
                motion = max(float(np.linalg.norm(np.array(sample["xyz"]) - origin)) for sample in tail)
                displacement_limit = 0.003 if hold_mode == "upright_tilt" else 0.005
                report["hold_stability"] = {
                    "mode": hold_mode,
                    "samples": len(tail),
                    "max_tilt_deg": max(angles) if hold_mode == "upright_tilt" else None,
                    "max_displacement_m": motion,
                    "limits": ({"tilt_deg": 15, "displacement_m": displacement_limit}
                               if hold_mode == "upright_tilt" else
                               {"displacement_m": displacement_limit}),
                }
                if hold_mode == "upright_tilt":
                    report["success"] = bool(report["success"] and max(angles) < 15 and motion < displacement_limit)
                else:
                    report["success"] = bool(report["success"] and motion < displacement_limit)
            else:
                report["hold_stability"] = {"mode": hold_mode, "samples": len(tail),
                                            "error": "insufficient hold samples"}
                report["success"] = False
        if hold_mode == "upright_tilt":
            # Historic field name kept for existing analysis scripts.
            report["cylinder_hold_check"] = report["hold_stability"]
        full_state = n.service(
            GetEntityState, "/gazebo/get_entity_state",
            GetEntityState.Request(name="cylinder", reference_frame="world"))
        if full_state.success:
            p, q = full_state.state.pose.position, full_state.state.pose.orientation
            report["object_pose"] = {
                "position_xyz": [p.x, p.y, p.z],
                "orientation_xyzw": [q.x, q.y, q.z, q.w],
            }
        finger_end = None
        if report.get("executed_end_states"):
            last_state = report["executed_end_states"][-1]
            if "gripper_joint1" in last_state and "gripper_joint2" in last_state:
                finger_end = [last_state["gripper_joint1"], last_state["gripper_joint2"]]
        grasp_state, grasp_details = classify_grasp(
            report["finger_contacts"], finger_end, close_width,
            lift_m=report["object_lift_m"], both_phases=True)
        report["grasp_state"] = grasp_state
        report["grasp_state_details"] = grasp_details
        # Payload prediction is no longer needed once the grasp is decided.
        detach = ApplyPlanningScene.Request()
        detach.scene.is_diff = True
        detach.scene.robot_state.is_diff = True
        detach.scene.robot_state.attached_collision_objects.append(
            AttachedCollisionObject(
                link_name="gripper_link2",
                object=CollisionObject(id="target", operation=CollisionObject.REMOVE),
            )
        )
        n.service(ApplyPlanningScene, "/apply_planning_scene", detach)
        mark("evaluated")
        report["stage_timings_s"] = stage_timings
        # Success carries an explicit null stage; failures keep whatever the
        # exception handler recorded.
        report["failure_stage"] = None
        # Full-process success needs the physical grasp, hold stability AND
        # settled joint feedback; each component is reported separately.
        report["execution_complete"] = bool(
            report["success"] and report.get("lift_feedback_settled"))
        report["status"] = "completed"
    except Exception as e:
        report.update(status="failed", success=False, reason=str(e))
        report.setdefault("failure_stage", main_stage[0])
        report.setdefault("physical_grasp_success", False)
        report["execution_complete"] = False
        report["stage_timings_s"] = stage_timings
        print(str(e), flush=True)
    finally:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        n.destroy_node()
        rclpy.shutdown()
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
