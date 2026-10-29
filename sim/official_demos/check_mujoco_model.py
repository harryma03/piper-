#!/usr/bin/env python3
"""Load the unchanged official Piper MuJoCo scene and check its gripper."""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/agx_arm_sim/mujoco/agilex_arm/agilex_piper/scene.xml"
OUT = ROOT / "records/official_demo_tests"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"status": "starting", "model": str(SOURCE), "mujoco": mujoco.__version__,
              "scope": "model dynamics and empty gripper; no grasped object", "stages": []}
    try:
        model = mujoco.MjModel.from_xml_path(str(SOURCE))
        data = mujoco.MjData(model)
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home)
        grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ("joint7", "joint8")]
        qpos_ids = [model.jnt_qposadr[j] for j in joint_ids]
        result.update(nq=model.nq, nv=model.nv, nu=model.nu, timestep_s=model.opt.timestep,
                      gripper_actuator_force_limits=model.actuator_forcerange[grip].tolist())
        body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("link7", "link8")]
        result["finger_collision_geoms"] = [
            {"geom_id": i, "friction_slide_torsion_roll": model.geom_friction[i].tolist()}
            for i in range(model.ngeom) if model.geom_bodyid[i] in body_ids and model.geom_contype[i] != 0]
        for target in (0.0, 0.03, 0.0):
            data.ctrl[grip] = target
            for _ in range(round(3.0 / model.opt.timestep)):
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Simulation produced nonfinite state")
            observed = data.qpos[qpos_ids].tolist()
            stage = {"target_per_finger_m": target, "finger_positions_m": observed,
                     "actuator_force": float(data.actuator_force[grip]),
                     "passed": abs(observed[0] - target) < 0.003 and abs(observed[1] + target) < 0.003}
            result["stages"].append(stage)
            if not stage["passed"]:
                raise RuntimeError(f"Empty gripper did not track target: {stage}")
        result["status"] = "passed"
        try:
            from PIL import Image
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.1, 0, 0.25]
                camera.distance = 1.25
                camera.azimuth = 135
                camera.elevation = -25
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(OUT / "mujoco_piper.png")
            result["image"] = str(OUT / "mujoco_piper.png")
        except Exception as exc:
            result["render_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (OUT / "mujoco_piper_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
