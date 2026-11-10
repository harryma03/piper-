#!/usr/bin/env python3
"""Orchestrate the active-perception comparison experiment.

Per scene and mode: start the isolated Gazebo scene, start the resident
perception service, optionally start the independent disturbance injector,
run run_closed_loop.py, collect the result, tear everything down.

Three modes share one code path and one code version:
  single    - one viewpoint, one plan, no mid-approach updates
  multiview - planner-driven extra viewpoints fused before planning
  closedloop- multi-view + segmented approach with per-segment perception
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stop_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def cleanup_strays():
    subprocess.run(['pkill', '-f', 'gz[s]erver'], check=False)
    subprocess.run(['pkill', '-f', 'move_[g]roup'], check=False)
    subprocess.run(['pkill', '-f', 'robot_state_[p]ublisher'], check=False)
    subprocess.run(['pkill', '-f', 'perception_[s]ervice'], check=False)
    subprocess.run(['pkill', '-f', 'inject_[d]isturbance'], check=False)
    subprocess.run(['pkill', '-f', 'run_[c]losed_loop'], check=False)
    time.sleep(2)


def run_trial(scene, mode, out_dir, base_env, keep_last=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_gazebo = proc_service = proc_injector = None
    trial = {'scene': scene.get('name'), 'mode': mode, 'status': 'setup_failed'}
    try:
        scene_env = dict(base_env)
        scene_env.update({
            'PIPER_SIM_OBJECT_SHAPE': scene['shape'],
            'PIPER_SIM_OBJECT_DIMS': ','.join(f'{float(v):.4f}' for v in scene['dims']),
            'PIPER_SIM_OBJECT_X': f"{scene['xy'][0]:.4f}",
            'PIPER_SIM_OBJECT_Y': f"{scene['xy'][1]:.4f}",
            'PIPER_SIM_OBJECT_YAW_DEG': f"{float(scene.get('yaw', 0)):.4f}",
            'PIPER_SIM_OCCLUDER': '1' if scene.get('occluder') else '0',
        })
        with (out_dir / 'scene.log').open('w') as log:
            proc_gazebo = subprocess.Popen(
                ['bash', 'scripts/start_gazebo_grasp.sh'], cwd=ROOT, env=scene_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 75
        ready = False
        while time.monotonic() < deadline:
            if proc_gazebo.poll() is not None:
                raise RuntimeError('Gazebo startup died')
            if 'Isolated contact simulation ready' in \
                    (out_dir / 'scene.log').read_text(errors='replace'):
                ready = True
                break
            time.sleep(0.3)
        if not ready:
            raise RuntimeError('Gazebo startup timeout')
        trial['status'] = 'perception_startup'

        service_env = dict(base_env)
        proc_service = subprocess.Popen(
            [sys.executable, 'sim/gazebo/grasp/perception_service.py'],
            cwd=ROOT, env=service_env,
            stdout=(out_dir / 'perception.log').open('w'),
            stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(12)  # worker model load (resident afterwards)

        if scene.get('disturbance'):
            injector_cmd = [sys.executable, 'scripts/inject_disturbance.py',
                            '--schedule', json.dumps(scene['disturbance'])]
            proc_injector = subprocess.Popen(
                injector_cmd, cwd=ROOT, env=base_env,
                stdout=(out_dir / 'injector.log').open('w'),
                stderr=subprocess.STDOUT, start_new_session=True)

        trial['status'] = 'running'
        cmd = [sys.executable, 'sim/gazebo/grasp/run_closed_loop.py',
               '--mode', mode, '--out', str(out_dir),
               '--max-extra-views', str(scene.get('max_extra_views', 2))]
        result = subprocess.run(cmd, cwd=ROOT, env=base_env,
                                stdout=(out_dir / 'executor.log').open('w'),
                                stderr=subprocess.STDOUT, timeout=420)
        result_file = out_dir / 'closed_loop_result.json'
        if result_file.is_file():
            trial = json.loads(result_file.read_text())
            trial['scene'] = scene.get('name')
            trial['mode'] = mode
            trial['returncode'] = result.returncode
        else:
            trial = {'scene': scene.get('name'), 'mode': mode,
                     'status': 'no_result_file', 'returncode': result.returncode}
        return trial
    except subprocess.TimeoutExpired:
        trial['status'] = 'executor_timeout'
        return trial
    except Exception as exc:  # noqa: BLE001
        trial['reason'] = str(exc)
        return trial
    finally:
        stop_proc(proc_injector)
        stop_proc(proc_service)
        if not keep_last:
            stop_proc(proc_gazebo)


def main():
    parser = __import__('argparse').ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--modes', default='single,multiview,closedloop')
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    scenes = json.loads(args.scenes.read_text())
    if isinstance(scenes, dict):
        scenes = scenes['scenes']
    modes = args.modes.split(',')
    args.out_root.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ)
    base_env.update({
        'ROS_DOMAIN_ID': '183', 'ROS_LOCALHOST_ONLY': '1',
        'GAZEBO_MASTER_URI': 'http://127.0.0.1:11371',
        'PYTHONPATH': f'{ROOT}/src/piperx_lab:{os.environ.get("PYTHONPATH", "")}',
        'PIPER_SIM_RGBD': '1', 'PIPER_SIM_HEADLESS': '1',
        'PIPER_SIM_ARM_PID': '1', 'PIPER_SIM_CONTROL_RATE': '1000',
        'PIPER_SIM_MIN_CONTACT_DEPTH': '0', 'PIPER_SIM_FINGER_KP': '300',
        'PIPER_SIM_FRICTION': '2', 'PIPER_SIM_OBJECT_MASS': '0.06',
    })
    summary_path = args.out_root / 'experiment_summary.json'
    all_trials = []
    try:
        for scene in scenes[:args.limit] if args.limit else scenes:
            for mode in modes:
                label = f"{scene.get('name', 'scene')}_{mode}"
                out_dir = args.out_root / label
                print(f'=== {label} ===', flush=True)
                cleanup_strays()
                trial = run_trial(scene, mode, out_dir, base_env)
                all_trials.append(trial)
                print(json.dumps({k: trial.get(k) for k in
                                  ('scene', 'mode', 'status', 'success',
                                   'execution_complete', 'failure_stage')}), flush=True)
                (args.out_root / 'experiment_summary.json').write_text(
                    json.dumps(all_trials, indent=1, default=str) + '\n')
    except KeyboardInterrupt:
        all_trials.append({'status': 'interrupted'})
    finally:
        cleanup_strays()
        summary_path.write_text(json.dumps(all_trials, indent=1, default=str) + '\n')
        print(f'saved {summary_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
