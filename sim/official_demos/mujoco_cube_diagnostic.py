#!/usr/bin/env python3
"""Add a free cube to the official Piper MuJoCo scene for grasp diagnostics.

The cube and control sequence are local additions, not an upstream demo.
No attachment or equality constraint is added between robot and cube.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/agx_arm_sim/mujoco/agilex_arm/agilex_piper/scene.xml"


def idx(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Missing model element {name}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--gripper-kp", type=float, default=40.0)
    parser.add_argument("--mass", type=float, default=0.08)
    parser.add_argument("--offset-x", type=float, default=0.0, help="Initial cube displacement in world x, metres")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Initial cube displacement in world y, metres")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Initial cube yaw in degrees")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.friction, args.gripper_kp, args.mass)):
        parser.error("Parameters must be finite and positive")
    if not all(math.isfinite(x) for x in (args.offset_x, args.offset_y, args.yaw_deg)):
        parser.error("Offsets and yaw must be finite")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "starting", "scope": "custom cube test using unchanged official MuJoCo robot",
              "source_scene": str(SOURCE), "cube_size_m": 0.04, "cube_mass_kg": args.mass,
              "sliding_friction_fingers_and_cube": args.friction, "gripper_kp": args.gripper_kp,
              "initial_offset_x_m": args.offset_x, "initial_offset_y_m": args.offset_y,
              "initial_yaw_deg": args.yaw_deg,
              "cube_attached_to_robot": False, "stages": {}}
    try:
        # Determine a reproducible initial cube position from the original model.
        reference = mujoco.MjModel.from_xml_path(str(SOURCE))
        reference_data = mujoco.MjData(reference)
        home_id = idx(reference, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(reference, reference_data, home_id)
        for joint_name, value in (("joint7", 0.03), ("joint8", -0.03)):
            j = idx(reference, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            reference_data.qpos[reference.jnt_qposadr[j]] = value
        mujoco.mj_forward(reference, reference_data)
        fingers = [idx(reference, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("link7", "link8")]
        # Use the first collision box on each finger; mesh geoms are visual-only.
        collision_ids = [[i for i in range(reference.ngeom) if reference.geom_bodyid[i] == body
                          and reference.geom_contype[i] != 0] for body in fingers]
        center = (reference_data.geom_xpos[collision_ids[0][0]] + reference_data.geom_xpos[collision_ids[1][0]]) / 2
        center += [args.offset_x, args.offset_y, 0.0]
        table_top = float(center[2] - 0.020)

        spec = mujoco.MjSpec.from_file(str(SOURCE))
        spec.worldbody.add_geom(name="test_support", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[float(center[0]), float(center[1]), table_top - 0.01],
                                size=[0.12, 0.12, 0.01], rgba=[0.55, 0.55, 0.55, 1])
        cube_spec = spec.worldbody.add_body(name="test_cube", pos=center.tolist())
        cube_spec.add_freejoint(name="cube_free")
        cube_spec.add_geom(name="test_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[0.02, 0.02, 0.02], mass=args.mass,
                           friction=[args.friction, 0.005, 0.0001], rgba=[0.9, 0.12, 0.08, 1])
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, idx(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        cube_joint = idx(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = model.jnt_qposadr[cube_joint]
        data.qpos[cube_qpos:cube_qpos + 3] = center
        half_yaw = math.radians(args.yaw_deg) / 2
        data.qpos[cube_qpos + 3:cube_qpos + 7] = [math.cos(half_yaw), 0, 0, math.sin(half_yaw)]
        for joint_name, value in (("joint7", 0.03), ("joint8", -0.03)):
            j = idx(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[model.jnt_qposadr[j]] = value
        actuator = idx(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        arm_actuator = idx(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint3")
        data.ctrl[actuator] = 0.03
        original_arm_target = float(data.ctrl[arm_actuator])
        model.actuator_gainprm[actuator, 0] = args.gripper_kp
        model.actuator_biasprm[actuator, 1] = -args.gripper_kp
        finger_bodies = [idx(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("link7", "link8")]
        tool_body = idx(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        finger_geoms = {i for i in range(model.ngeom)
                        if model.geom_bodyid[i] in finger_bodies and model.geom_contype[i] != 0}
        first_finger_geom = [min(i for i in finger_geoms if model.geom_bodyid[i] == body)
                             for body in finger_bodies]
        for i in finger_geoms:
            model.geom_friction[i, 0] = args.friction
        cube_geom = idx(model, mujoco.mjtObj.mjOBJ_GEOM, "test_cube_geom")
        cube_body = idx(model, mujoco.mjtObj.mjOBJ_BODY, "test_cube")
        result["initial_cube_position_m"] = center.tolist()
        result["actuator_force_limits"] = model.actuator_forcerange[actuator].tolist()
        result["timestep_s"] = float(model.opt.timestep)
        mujoco.mj_forward(model, data)

        def snapshot(phase, step):
            normal = [0.0, 0.0]
            tangential = [0.0, 0.0]
            count = [0, 0]
            for ci in range(data.ncon):
                contact = data.contact[ci]
                a, b = int(contact.geom1), int(contact.geom2)
                other = b if a == cube_geom else a if b == cube_geom else -1
                if other not in finger_geoms:
                    continue
                fi = finger_bodies.index(int(model.geom_bodyid[other]))
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                normal[fi] += abs(float(force[0]))
                tangential[fi] += float(np.linalg.norm(force[1:3]))
                count[fi] += 1
            center_world = data.xpos[cube_body].copy()
            mid_fingers = (data.geom_xpos[first_finger_geom[0]] + data.geom_xpos[first_finger_geom[1]]) / 2
            relative = data.xmat[tool_body].reshape(3, 3).T @ (center_world - mid_fingers)
            quat = data.xquat[cube_body]
            positions = [float(data.qpos[model.jnt_qposadr[idx(model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
                         for name in ("joint7", "joint8")]
            return {"phase": phase, "step": step, "time_s": float(data.time),
                    "cube_x_m": float(center_world[0]), "cube_y_m": float(center_world[1]),
                    "cube_z_m": float(center_world[2]), "cube_lift_m": float(center_world[2] - center[2]),
                    "cube_rel_tool_x_m": float(relative[0]), "cube_rel_tool_y_m": float(relative[1]),
                    "cube_rel_tool_z_m": float(relative[2]),
                    "cube_quat_w": float(quat[0]), "cube_quat_x": float(quat[1]),
                    "cube_quat_y": float(quat[2]), "cube_quat_z": float(quat[3]),
                    "finger1_position_m": positions[0], "finger2_position_m": positions[1],
                    "finger1_normal_sum_n": normal[0], "finger2_normal_sum_n": normal[1],
                    "finger1_tangent_magnitudes_sum_n": tangential[0],
                    "finger2_tangent_magnitudes_sum_n": tangential[1],
                    "finger1_contacts": count[0], "finger2_contacts": count[1],
                    "actuator_force_n": float(data.actuator_force[actuator])}

        samples = []
        phases = (("settle_open", 0.5), ("close", 2.0), ("lift", 3.0), ("hold", 1.0))
        for phase, duration in phases:
            steps = round(duration / model.opt.timestep)
            for step in range(steps):
                if phase != "settle_open":
                    data.ctrl[actuator] = 0.0
                if phase == "lift":
                    data.ctrl[arm_actuator] = original_arm_target - 0.14 * (step + 1) / steps
                elif phase == "hold":
                    data.ctrl[arm_actuator] = original_arm_target - 0.14
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Nonfinite simulation state")
                if step % 5 == 0 or step == steps - 1:
                    samples.append(snapshot(phase, step))
            result["stages"][phase] = snapshot(phase, steps - 1)
        with (args.output / "trajectory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        hold = [s for s in samples if s["phase"] == "hold" and s["time_s"] >= data.time - 0.5]
        sustained = [s for s in hold if s["cube_lift_m"] > 0.05 and
                     s["finger1_normal_sum_n"] > 0.01 and s["finger2_normal_sum_n"] > 0.01]
        def relative_position(sample):
            return np.array([sample[f"cube_rel_tool_{axis}_m"] for axis in "xyz"])
        hold_origin = relative_position(hold[0]) if hold else np.zeros(3)
        result["relative_motion_during_final_half_second_m"] = max(
            (float(np.linalg.norm(relative_position(s) - hold_origin)) for s in hold), default=None)
        close_sample = result["stages"]["close"]
        result["relative_displacement_close_to_final_m"] = float(
            np.linalg.norm(relative_position(samples[-1]) - relative_position(close_sample)))
        result["peak_lift_m"] = max(s["cube_lift_m"] for s in samples)
        result["final_lift_m"] = samples[-1]["cube_lift_m"]
        result["final_half_second_samples"] = len(hold)
        result["final_half_second_two_finger_lifted_samples"] = len(sustained)
        result["grasp_held"] = (len(hold) > 0 and len(sustained) == len(hold)
                                and result["relative_motion_during_final_half_second_m"] < 0.005)
        result["status"] = "completed"
        try:
            if args.no_render:
                return
            from PIL import Image
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.40, 0, 0.35]
                camera.distance = 0.75
                camera.azimuth = 120
                camera.elevation = -20
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(args.output / "final.png")
        except Exception as exc:
            result["render_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
