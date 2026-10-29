#!/usr/bin/env python3
"""Replay an Isaac Lab recording in a MuJoCo window using the same URDF.

This is a visualization of recorded states, not a new physics simulation.
"""
import argparse
import json
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROBOT = ROOT / "third_party/IsaacLab_Data_Collection/source/agx_teleop/agx_description/agx_arm_urdf/piper/urdf/piper_gripper.urdf"
DEFAULT_DATASET = ROOT / "records/isaac_stack/upstream_unmodified/dataset.hdf5"


def index(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Missing {name} in replay model")
    return value


def load_model():
    # USD was generated from this URDF. MuJoCo does not load DAE meshes, so
    # use the corresponding STL meshes provided by the same source model.
    root = ET.parse(ROBOT).getroot()
    mujoco_tag = ET.SubElement(root, "mujoco")
    ET.SubElement(mujoco_tag, "compiler", {
        "discardvisual": "false", "fusestatic": "false", "balanceinertia": "true",
        "meshdir": str(ROBOT.parent.parent / "meshes"),
    })
    for mesh in root.findall(".//mesh"):
        name = Path(mesh.get("filename")).stem + ".stl"
        if not (ROBOT.parent.parent / "meshes" / name).is_file():
            raise FileNotFoundError(name)
        mesh.set("filename", name)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.body("gripper_base").add_site(name="isaac_tcp", pos=[0, 0, 0.14], size=[0.002, 0.002, 0.002])
    spec.worldbody.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1])
    spec.worldbody.add_geom(name="display_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[2, 2, 0.05], pos=[0, 0, -0.11], rgba=[0.24, 0.29, 0.34, 1])
    spec.worldbody.add_geom(name="display_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[0.4, 0, -0.0231], size=[0.35, 0.30, 0.02],
                            rgba=[0.55, 0.55, 0.55, 1])
    colors = ([0.1, 0.3, 0.9, 1], [0.85, 0.1, 0.1, 1], [0.1, 0.7, 0.2, 1])
    for n, rgba in enumerate(colors, 1):
        body = spec.worldbody.add_body(name=f"replay_cube_{n}")
        body.add_freejoint(name=f"replay_cube_joint_{n}")
        body.add_geom(name=f"replay_cube_geom_{n}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.0234] * 3, mass=0.02, rgba=rgba)
    model = spec.compile()
    # Highlight the same-source robot while leaving the replay cubes colored.
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_rgba[i] = [0.75, 0.78, 0.80, 1]
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", default="demo_0")
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as source:
        env_args = json.loads(source["data"].attrs["env_args"])
        episode = source[f"data/{args.episode}"]
        joints = episode["states/articulation/robot/joint_position"][:]
        cubes = np.stack([episode[f"states/rigid_object/cube_{n}/root_pose"][:] for n in (1, 2, 3)], axis=1)
        eef_before_step = episode["obs/eef_pos"][:]
    dt = env_args["sim_args"]["dt"] * env_args["sim_args"]["decimation"]
    if len(joints) < 2 or joints.shape[1] != 8 or cubes.shape != (len(joints), 3, 7):
        raise ValueError("Recording has an unexpected shape")

    model = load_model()
    data = mujoco.MjData(model)
    arm_q = [model.jnt_qposadr[index(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
             for name in [*(f"joint{i}" for i in range(1, 7)), "gripper_joint1", "gripper_joint2"]]
    cube_q = [model.jnt_qposadr[index(model, mujoco.mjtObj.mjOBJ_JOINT, f"replay_cube_joint_{i}")]
              for i in (1, 2, 3)]
    tcp = index(model, mujoco.mjtObj.mjOBJ_SITE, "isaac_tcp")
    deviations = []
    # The observation is pre-step; the recorded robot state is post-step.
    for i in range(1, len(joints)):
        data.qpos[arm_q] = joints[i - 1]
        mujoco.mj_forward(model, data)
        deviations.append(np.linalg.norm(data.site_xpos[tcp] - eef_before_step[i]))
    max_error = float(np.max(deviations))
    print(f"Maximum TCP mismatch to recorded Isaac pose: {max_error * 1000:.4f} mm", flush=True)
    if not math.isfinite(max_error) or max_error > 0.002:
        raise RuntimeError("Replay model does not match the recorded end-effector pose")

    state = {"paused": False, "restart": False}

    def key_callback(keycode):
        if keycode == 32:  # Space
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["restart"] = True

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0.38, 0.0, 0.16]
        viewer.cam.distance = 1.05
        viewer.cam.azimuth = 132
        viewer.cam.elevation = -20
        frame = 0
        next_tick = time.monotonic()
        print("REPLAY_WINDOW_READY: Isaac recording in MuJoCo viewer", flush=True)
        while viewer.is_running():
            if state["restart"]:
                frame = 0
                state["restart"] = False
            if not state["paused"]:
                if frame == 0: print("Waiting for viewer lock", flush=True)
                with viewer.lock():
                    data.qpos[arm_q] = joints[frame]
                    for i, address in enumerate(cube_q):
                        data.qpos[address:address + 7] = cubes[frame, i]
                    mujoco.mj_forward(model, data)
                if frame == 0: print("Updated first frame", flush=True)
                frame = (frame + 1) % len(joints)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "Isaac Lab recording | MuJoCo replay",
                              "Space: pause / resume    R: restart    Drag: rotate    Scroll: zoom"))
            if frame == 1: print("Syncing first frame", flush=True)
            viewer.sync()
            if frame == 1: print("First frame synced", flush=True)
            next_tick += dt
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()


if __name__ == "__main__":
    main()
