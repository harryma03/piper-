#!/usr/bin/env python3
"""Bounded Piper cube-stack check using the upstream IK controller unchanged.

Run with .isaac-venv/bin/python. AppLauncher must initialize Isaac Sim before
any torch/omni/Isaac Lab task imports (the upstream module handles this).
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/IsaacLab_Data_Collection/scripts/tools/record_ik_stack.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "records/isaac_stack/trial")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=700, help="Maximum control steps per trial")
    parser.add_argument("--seed", type=int, default=42)
    args, launcher_args = parser.parse_known_args()
    if args.trials < 1 or args.max_steps < 1:
        parser.error("--trials and --max-steps must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    result = {"status": "starting", "upstream": str(UPSTREAM), "seed": args.seed, "trials": []}

    def save():
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(result_path)

    save()
    upstream = None
    env = None
    try:
        sys.argv = [str(UPSTREAM), "--task", "Isaac-Stack-Cube-Piper-IK-Rel-v0",
                    "--dataset_file", str(args.output / "upstream_dataset.hdf5"),
                    "--num_demos", "1", *launcher_args]
        spec = importlib.util.spec_from_file_location("piper_upstream_ik_stack", UPSTREAM)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        torch = upstream.torch
        output_dir, output_name = upstream.setup_output_directories()
        cfg, success_term = upstream.create_environment_config(output_dir, output_name)
        cfg.seed = args.seed
        env = upstream.create_environment(cfg)
        gripper_ids, gripper_names = env.scene["robot"].find_joints(["gripper_joint.*"])
        result["gripper_joint_names"] = gripper_names
        result["control_dt_s"] = env.step_dt
        result["status"] = "running"
        save()

        columns = ["trial", "step", "sim_time_s", "command_grip", "upstream_success",
                   "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"]
        for i in range(1, 4):
            columns += [f"cube{i}_{axis}" for axis in ("x", "y", "z")]
        columns += [f"position_{name}" for name in gripper_names]
        columns += [f"applied_effort_{name}" for name in gripper_names]
        with (args.output / "trajectory.csv").open("w", newline="") as stream, torch.inference_mode():
            writer = csv.writer(stream)
            writer.writerow(columns)
            for trial in range(args.trials):
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                if trial == 0:
                    result["physics"] = {}
                    for name in ("robot", "cube_1", "cube_2", "cube_3"):
                        asset = env.scene[name]
                        result["physics"][name] = {
                            "mass_kg": asset.root_physx_view.get_masses().cpu().tolist(),
                            "shape_material_static_dynamic_friction_restitution":
                                asset.root_physx_view.get_material_properties().cpu().tolist(),
                        }
                    save()
                controller = upstream.IKStackController(env)
                controller.plan()
                initial_z = [float(c.data.root_pos_w[0, 2]) for c in controller.cubes]
                metrics = {"trial": trial, "initial_z_m": initial_z,
                           "max_lift_m": [0.0] * 3, "max_closed_lift_streak_steps": [0] * 3,
                           "upstream_success": False, "stop_reason": "max_steps"}
                result["trials"].append(metrics)
                streak = [0] * 3
                success_streak = 0
                for step in range(args.max_steps):
                    if not upstream.simulation_app.is_running():
                        metrics["stop_reason"] = "app_stopped"
                        break
                    action = controller.compute_action(env)
                    _, _, terminated, truncated, _ = env.step(action)
                    # Auto-reset would hide the terminal state. Stop this trial immediately.
                    if bool(terminated[0]) or bool(truncated[0]):
                        metrics["stop_reason"] = "environment_terminated_or_reset"
                        break
                    success = (bool(success_term.func(env, **success_term.params)[0])
                               if success_term is not None else False)
                    success_streak = success_streak + 1 if success else 0
                    grip = float(action[0, -1])
                    ee = controller.ee_frame.data.target_pos_w[0, 0].cpu().tolist()
                    quat = controller.ee_frame.data.target_quat_w[0, 0].cpu().tolist()
                    cube_positions = [c.data.root_pos_w[0].cpu().tolist() for c in controller.cubes]
                    robot = env.scene["robot"]
                    positions = robot.data.joint_pos[0, gripper_ids].cpu().tolist()
                    # The implicit actuator reports a clipped PD estimate, not a measured
                    # joint force or finger contact force.
                    efforts = robot.data.applied_torque[0, gripper_ids].cpu().tolist()
                    writer.writerow([trial, step, (step + 1) * env.step_dt, grip, int(success),
                                     *ee, *quat, *(v for pos in cube_positions for v in pos),
                                     *positions, *efforts])
                    for i, pos in enumerate(cube_positions):
                        lift = pos[2] - initial_z[i]
                        metrics["max_lift_m"][i] = max(metrics["max_lift_m"][i], lift)
                        # Height plus a close command alone does not prove a stable grasp.
                        streak[i] = streak[i] + 1 if grip < 0 and lift > 0.05 else 0
                        metrics["max_closed_lift_streak_steps"][i] = max(
                            metrics["max_closed_lift_streak_steps"][i], streak[i])
                    metrics["steps"] = step + 1
                    metrics["final_cube_positions_m"] = cube_positions
                    if step % 30 == 0:
                        stream.flush()
                        save()
                        print(f"[cube-check] trial={trial} step={step} z="
                              f"{[round(p[2], 4) for p in cube_positions]}", flush=True)
                    if success_streak >= 10:
                        metrics["upstream_success"] = True
                        metrics["stop_reason"] = "upstream_success"
                        # Preserve upstream dataset export behavior for a successful trial.
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
                        env.recorder_manager.export_episodes([0])
                        break
                    if controller.should_reset():
                        metrics["stop_reason"] = "trajectory_completed"
                        break
                save()
        result["status"] = "completed"
        save()
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0 if any(t["upstream_success"] for t in result["trials"]) else 2
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        save()
        raise
    finally:
        if env is not None:
            env.close()
        if upstream is not None and hasattr(upstream, "simulation_app"):
            upstream.simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
