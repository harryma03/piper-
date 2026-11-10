#!/usr/bin/env python3
"""Run independent, randomized RGB-D-to-contact Gazebo grasp trials.

The target position is passed only to the simulator setup. The observer and
planner get sensor observations instead; simulator truth is used for metrics.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=ROOT / 'records/gazebo_vision_grasp' / time.strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    if not 1 <= args.trials <= 100:
        parser.error('trials must be between 1 and 100')
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    summary = {'requested_trials': args.trials, 'seed': args.seed,
               'region_m': {'x': [0.40, 0.50], 'y': [-0.05, 0.05]},
               'truth_usage': 'scene setup and final evaluation only',
               'prior': 'upright 46 mm diameter, 80 mm height on plane z=-0.15 m',
               'trial_results': [], 'status': 'running'}
    base_env = os.environ.copy()
    base_env['ROS_DOMAIN_ID'] = '183'
    base_env['GAZEBO_MASTER_URI'] = 'http://127.0.0.1:11371'
    base_env['PYTHONPATH'] = str(ROOT/'src/piperx_lab') + os.pathsep + base_env.get('PYTHONPATH', '')
    # Trial coordinates are passed to the scene generator, never to observation or planning.
    for key in ('PIPER_SIM_OBJECT_X', 'PIPER_SIM_OBJECT_Y'):
        base_env.pop(key, None)
    base_env.update(PIPER_SIM_RGBD='1', PIPER_SIM_HEADLESS='1', PIPER_SIM_ARM_PID='1',
                    PIPER_SIM_CONTROL_RATE='1000', PIPER_SIM_MIN_CONTACT_DEPTH='0',
                    PIPER_SIM_FINGER_KP='300', PIPER_SIM_FRICTION='2',
                    PIPER_SIM_OBJECT_SHAPE='cylinder', PIPER_SIM_OBJECT_MASS='.06')
    summary_path = args.out/'summary.json'
    try:
        for i in range(args.trials):
            trial_dir = args.out / f'trial_{i+1:03d}'
            trial_dir.mkdir()
            x, y = rng.uniform(.40, .50), rng.uniform(-.05, .05)
            trial = {'trial': i+1, 'spawn_xy_m': [x, y], 'status': 'setup_failed', 'success': False}
            summary['trial_results'].append(trial)
            scene_env = dict(base_env, PIPER_SIM_OBJECT_X=str(x), PIPER_SIM_OBJECT_Y=str(y))
            scene = None
            print(f'Trial {i+1}/{args.trials}: starting isolated Gazebo', flush=True)
            try:
                with (trial_dir/'scene.log').open('w') as log:
                    scene = subprocess.Popen(['bash', 'scripts/start_gazebo_grasp.sh'], cwd=ROOT,
                                             env=scene_env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if scene.poll() is not None:
                        raise RuntimeError('Gazebo startup stopped; see scene.log')
                    if 'Isolated contact simulation ready' in (trial_dir/'scene.log').read_text(errors='replace'):
                        break
                    time.sleep(.25)
                else:
                    raise TimeoutError('Gazebo startup timed out')
                obs_file = trial_dir/'observation.json'
                with (trial_dir/'observation.log').open('w') as log:
                    observe = subprocess.run([sys.executable, 'sim/gazebo/grasp/observe_cylinder.py',
                                              '--out', str(obs_file), '--timeout', '15'],
                                             cwd=ROOT, env=base_env, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=25)
                if observe.returncode:
                    trial['status'] = 'recognition_failed'
                    if obs_file.exists():
                        trial['reason'] = json.loads(obs_file.read_text()).get('reason')
                    continue
                trial['status'] = 'grasp_failed'
                result_file = trial_dir/'result.json'
                with (trial_dir/'action.log').open('w') as log:
                    result = subprocess.run([sys.executable, 'sim/gazebo/grasp/run_grasp.py',
                        '--vision-observation', str(obs_file), '--grip-seconds', '8', '--hold-seconds', '5',
                        '--approach-scaling', '.03', '--lift-scaling', '.02', '--out', str(result_file)],
                        cwd=ROOT, env=base_env, stdout=log, stderr=subprocess.STDOUT, timeout=150)
                if result_file.exists():
                    data = json.loads(result_file.read_text())
                    trial.update(status='success' if data.get('success') else 'grasp_failed',
                                 success=bool(data.get('success')), stages=data.get('stages', []),
                                 vision_error_m=data.get('vision_error_m'),
                                 object_lift_m=data.get('object_lift_m'), reason=data.get('reason'),
                                 hold_check=data.get('cylinder_hold_check'))
                else:
                    trial['reason'] = f'Grasp runner exited {result.returncode} without a result'
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, ValueError) as exc:
                trial['reason'] = str(exc)
            finally:
                if scene is not None and scene.poll() is None:
                    os.killpg(scene.pid, signal.SIGINT)
                    try:
                        scene.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        os.killpg(scene.pid, signal.SIGKILL)
                        scene.wait(timeout=3)
                print(f"Trial {i+1}: {trial['status']}", flush=True)
                summary['completed_trials'] = len(summary['trial_results'])
                summary['success_count'] = sum(t['success'] for t in summary['trial_results'])
                summary['recognition_count'] = sum(t['status'] in ('success', 'grasp_failed') for t in summary['trial_results'])
                summary['success_rate'] = summary['success_count'] / len(summary['trial_results'])
                write_json(summary_path, summary)
        summary['status'] = 'completed'
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
        raise
    finally:
        write_json(summary_path, summary)
        print(f'Saved {summary_path}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
