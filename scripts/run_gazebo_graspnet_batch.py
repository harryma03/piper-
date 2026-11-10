#!/usr/bin/env python3
"""Run randomized simulated RGB-D GraspNet-to-contact grasp trials.

Each scene position is generated anew.  The spawn coordinate is used only to
configure the simulator and evaluate the final result; perception receives
the simulated RGB-D stream and image-time transforms.
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
GRASPNET_PYTHON = ROOT / '.graspnet-venv/bin/python'
CHECKPOINT = ROOT / 'third_party/graspnet-baseline/checkpoints/checkpoint-rs.tar'
# make_scene regenerates this file for every trial before capture runs.
SCENE_METADATA = Path('/tmp/piper_contact_grasp/scene_parameters.json')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def run_logged(command, log_path, env, timeout, acceptable=(0,)):
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode not in acceptable:
        raise RuntimeError(f'{Path(command[1] if len(command) > 1 else command[0]).name} '
                           f'exited {result.returncode}; see {log_path}')
    return result.returncode


def stop_scene(scene):
    if scene is None or scene.poll() is not None:
        return
    os.killpg(scene.pid, signal.SIGINT)
    try:
        scene.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(scene.pid, signal.SIGKILL)
        scene.wait(timeout=3)


GRASPABLE_DIM_RANGE_M = (0.02, 0.06)
GRASPABLE_HEIGHT_RANGE_M = (0.03, 0.09)


def sample_object_dims(rng, shape):
    """Random graspable dimensions for pilot scenes; 70 mm opening is the limit."""
    low, high = GRASPABLE_DIM_RANGE_M
    hlow, hhigh = GRASPABLE_HEIGHT_RANGE_M
    if shape == 'cylinder':
        return [round(rng.uniform(low, high), 4), round(rng.uniform(hlow, hhigh), 4)]
    if shape == 'sphere':
        return [round(rng.uniform(0.025, high), 4)]
    if shape == 'bar':
        return [round(rng.uniform(0.05, 0.09), 4), round(rng.uniform(0.02, 0.03), 4),
                round(rng.uniform(0.02, 0.03), 4)]
    return [round(rng.uniform(low, high), 4), round(rng.uniform(low, high), 4),
            round(rng.uniform(hlow, hhigh), 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=86)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--xy', action='append', metavar='X,Y',
                        help='Retest explicit base-frame positions; may be repeated')
    parser.add_argument('--roi-paddings', default='20,30,40,60,80')
    parser.add_argument('--symmetry-samples', type=int, default=16)
    parser.add_argument('--top-models', type=int, default=2)
    parser.add_argument('--object-shape', choices=('cylinder', 'cube', 'sphere', 'box', 'bar', 'mesh'),
                        default='cylinder', help='Simulated scene object shape')
    parser.add_argument('--object-dims', default='',
                        help='Comma-separated object dimensions in metres; "auto" samples graspable sizes per trial')
    parser.add_argument('--object-yaw', default='0',
                        help='Object yaw in degrees, or "auto" for a random yaw per trial')
    parser.add_argument('--object-mass', default='.06')
    parser.add_argument('--object-mesh', default='', help='STL/OBJ path for the mesh shape')
    parser.add_argument('--scenes-json', type=Path, default=None,
                        help='JSON list of per-trial scene params (shape/dims/yaw/xy/occluder/view); overrides --trials/--xy')
    parser.add_argument('--geometry-fallback', action='store_true',
                        help='Explicit config: append geometric candidates when network candidates all fail (stats reported separately)')
    parser.add_argument('--geometry-candidates', action='store_true',
                        help='Fusion mode: geometric antipodal candidates run alongside network candidates')
    parser.add_argument('--observation-mode', choices=('generic', 'cylinder', 'cube'),
                        default='generic', help='generic is shape-agnostic; cylinder/cube are legacy regression fits')
    parser.add_argument('--gui', action='store_true', help='Show the Gazebo client for each trial')
    parser.add_argument('--keep-open-last', action='store_true',
                        help='Keep the final Gazebo scene open after its trial')
    args = parser.parse_args()
    if not 1 <= args.trials <= 100:
        parser.error('trials must be in 1..100')
    fixed_xy = []
    scene_specs = None
    if args.scenes_json is not None:
        try:
            scene_payload = json.loads(args.scenes_json.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f'cannot read --scenes-json: {exc}')
        if isinstance(scene_payload, dict) and isinstance(scene_payload.get('scenes'), list):
            scene_payload = scene_payload['scenes']
        scene_specs = scene_payload
        if (not isinstance(scene_specs, list) or not 1 <= len(scene_specs) <= 200 or
                not all(isinstance(item, dict) for item in scene_specs)):
            parser.error('--scenes-json must be a JSON list of objects (1..200)')
        for spec in scene_specs:
            if 'xy' not in spec or 'shape' not in spec:
                parser.error('each scene spec needs at least "shape" and "xy"')
    for value in args.xy or []:
        try:
            x_text, y_text = value.split(',', 1)
            point = float(x_text), float(y_text)
        except ValueError as exc:
            parser.error(f'invalid --xy {value!r}; expected X,Y')
        fixed_xy.append(point)
    if fixed_xy and len(fixed_xy) > 100:
        parser.error('at most 100 explicit positions are supported')
    if scene_specs is not None:
        trial_count = len(scene_specs)
    else:
        trial_count = len(fixed_xy) if fixed_xy else args.trials
    try:
        roi_paddings = [int(value) for value in args.roi_paddings.split(',')]
    except ValueError as exc:
        raise SystemExit(f'invalid --roi-paddings: {exc}') from exc
    if (not roi_paddings or any(not 0 <= value <= 120 for value in roi_paddings)
            or not 1 <= args.top_models <= 4
            or not 1 <= args.symmetry_samples <= 20
            or args.symmetry_samples * args.top_models > 40):
        parser.error('invalid ROI, model or symmetry sampling count')
    try:
        object_mass = float(args.object_mass)
    except ValueError:
        parser.error('invalid --object-mass')
    if not 0.001 <= object_mass <= 1:
        parser.error('object mass must be within [0.001, 1] kg')
    auto_dims = args.object_dims.strip().lower() == 'auto'
    auto_yaw = args.object_yaw.strip().lower() == 'auto'
    if args.object_mesh and args.object_shape != 'mesh':
        parser.error('--object-mesh applies only to the mesh shape')
    if args.object_shape == 'mesh' and not args.object_mesh:
        parser.error('the mesh shape requires --object-mesh')
    if args.observation_mode != 'generic' and args.object_shape not in ('cylinder', 'cube'):
        parser.error('legacy observation modes only exist for the fixed cylinder/cube scenes')
    if not GRASPNET_PYTHON.is_file() or not CHECKPOINT.is_file():
        raise SystemExit('GraspNet environment or checkpoint is missing')
    if args.out.exists():
        raise SystemExit(f'Output already exists: {args.out}')
    args.out.mkdir(parents=True)

    rng = random.Random(args.seed)
    base_env = os.environ.copy()
    base_env.update(
        ROS_DOMAIN_ID='183',
        ROS_LOCALHOST_ONLY='1',
        GAZEBO_MASTER_URI='http://127.0.0.1:11371',
        PYTHONPATH=str(ROOT / 'src/piperx_lab') + os.pathsep + base_env.get('PYTHONPATH', ''),
        PIPER_SIM_RGBD='1',
        PIPER_SIM_HEADLESS='0' if args.gui else '1',
        PIPER_SIM_ARM_PID='1',
        PIPER_SIM_CONTROL_RATE='1000',
        PIPER_SIM_MIN_CONTACT_DEPTH='0',
        PIPER_SIM_FINGER_KP='300',
        PIPER_SIM_FRICTION='2',
        PIPER_SIM_OBJECT_SHAPE=args.object_shape,
        PIPER_SIM_OBJECT_MASS=str(object_mass),
        PIPER_SIM_OBJECT_DIMS='' if auto_dims else args.object_dims,
        PIPER_SIM_OBJECT_YAW_DEG='0' if auto_yaw else args.object_yaw,
        PIPER_SIM_OBJECT_MESH=args.object_mesh,
    )
    if Path('/tmp/.X11-unix/X1').exists():
        base_env.setdefault('DISPLAY', ':1')
        authority = Path('/run/user/1000/gdm/Xauthority')
        if authority.exists():
            base_env.setdefault('XAUTHORITY', str(authority))

    summary = {
        'requested_trials': trial_count,
        'seed': args.seed,
        'fixed_xy_m': fixed_xy or None,
        'region_m': {'x': [0.40, 0.50], 'y': [-0.05, 0.05]},
        'object_shape': args.object_shape,
        'object_dims': args.object_dims or None,
        'object_yaw_deg': args.object_yaw,
        'object_mass_kg': object_mass,
        'observation_mode': args.observation_mode,
        'geometry_fallback': args.geometry_fallback,
        'geometry_candidates': args.geometry_candidates,
        'success_definitions': {
            'physical_grasp_success': ('both finger contacts in closing and lifting AND '
                                       'lift > 0.07 m AND xy offset < 0.06 m'),
            'success': 'physical_grasp_success AND hold-stability gate (mode per observation)',
            'motion_feedback_settled': 'lift trajectory joint feedback reached command within 0.015',
            'execution_complete': 'success AND motion_feedback_settled; trial status uses this',
        },
        'object': (f'{args.object_shape} scene object; '
                   f'mass {object_mass} kg; dims {args.object_dims or "shape defaults"}; '
                   f'yaw {args.object_yaw} deg'),
        'pipeline': ('Gazebo RGB-D -> generic observation (color mask, support plane, '
                     'cluster, convex hull) -> GraspNet -> shared candidate filter -> '
                     'symmetry augmentation -> MoveIt convex-hull collision -> contact grasp'
                     if args.observation_mode == 'generic' else
                     'Gazebo RGB-D -> GraspNet -> Piper candidate filter -> '
                     'cylinder symmetry augmentation -> MoveIt -> contact grasp'),
        'truth_usage': 'scene setup and final evaluation only',
        'roi_paddings_px': roi_paddings,
        'symmetry_samples': args.symmetry_samples,
        'top_models': args.top_models,
        'trial_results': [],
        'status': 'running',
    }
    summary_path = args.out / 'summary.json'
    write_json(summary_path, summary)

    try:
        for index in range(trial_count):
            trial_dir = args.out / f'trial_{index + 1:03d}'
            trial_dir.mkdir()
            occluder = False
            view = 0
            if scene_specs is not None:
                spec = scene_specs[index]
                x, y = (float(v) for v in spec['xy'])
                if not (0.3 <= x <= 0.55 and abs(y) <= 0.12):
                    parser.error(f'scene spec {index + 1}: xy outside the supported test area')
                shape_for_trial = spec.get('shape', args.object_shape)
                dims_value = spec.get('dims')
                if dims_value == 'auto':
                    auto_dims, dims_text = True, ''
                elif isinstance(dims_value, list):
                    dims_text = ','.join(f'{float(v):.4f}' for v in dims_value)
                else:
                    dims_text = args.object_dims
                yaw_value = spec.get('yaw')
                if yaw_value is not None:
                    yaw_text = f'{float(yaw_value):.2f}'
                else:
                    yaw_text = ('0' if not auto_yaw else f'{rng.uniform(-180, 180):.2f}')
                occluder = bool(spec.get('occluder', False))
                view = int(spec.get('view', 0))
            else:
                shape_for_trial = args.object_shape
                if fixed_xy:
                    x, y = fixed_xy[index]
                else:
                    x = rng.uniform(0.40, 0.50)
                    y = rng.uniform(-0.05, 0.05)
                dims_text = (args.object_dims if not auto_dims else
                             ','.join(f'{value:.4f}' for value in sample_object_dims(rng, shape_for_trial)))
                yaw_text = (args.object_yaw if not auto_yaw else f'{rng.uniform(-180, 180):.2f}')
            trial = {
                'trial': index + 1,
                'status': 'setup_failed',
                'success': False,
                'spawn_xy_m': [x, y],
                'object_shape': shape_for_trial,
                'object_dims_scene_m': [float(value) for value in dims_text.split(',')] if dims_text else None,
                'object_yaw_deg': float(yaw_text),
                'occluder': occluder,
                'view': view,
                'segmentation_success': False,
                'collision_geometry_success': False,
                'candidate_count': None,
                'planning_success': False,
                'physical_grasp_success': False,
                'motion_feedback_settled': None,
                'max_object_lift_m': None,
                'hold_stability': None,
                'failure_stage': 'setup',
                'failure_reason': None,
                'object_dimensions_m': None,
                'object_pose': None,
            }
            summary['trial_results'].append(trial)
            scene_env = dict(base_env, PIPER_SIM_OBJECT_X=str(x), PIPER_SIM_OBJECT_Y=str(y),
                             PIPER_SIM_OBJECT_SHAPE=shape_for_trial,
                             PIPER_SIM_OBJECT_DIMS=dims_text, PIPER_SIM_OBJECT_YAW_DEG=yaw_text,
                             PIPER_SIM_OCCLUDER='1' if occluder else '0',
                             PIPER_SIM_VIEW=str(view))
            scene = None
            print(f'Trial {index + 1}/{trial_count}: Gazebo startup', flush=True)
            try:
                with (trial_dir / 'scene.log').open('w') as log:
                    scene = subprocess.Popen(
                        ['bash', 'scripts/start_gazebo_grasp.sh'],
                        cwd=ROOT, env=scene_env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                deadline = time.monotonic() + 75
                while time.monotonic() < deadline:
                    if scene.poll() is not None:
                        raise RuntimeError('Gazebo startup stopped; see scene.log')
                    if 'Isolated contact simulation ready' in (
                            trial_dir / 'scene.log').read_text(errors='replace'):
                        break
                    time.sleep(0.25)
                else:
                    raise TimeoutError('Gazebo startup timed out')

                frame = trial_dir / 'frame.npz'
                run_logged(
                    [sys.executable, 'sim/gazebo/grasp/capture_graspnet.py',
                     '--out', str(frame), '--timeout', '35',
                     '--observation-mode', args.observation_mode],
                    trial_dir / 'capture.log', base_env, 45,
                )
                observation_data = json.loads(
                    (trial_dir / 'frame_observation.json').read_text())
                trial['segmentation_success'] = observation_data.get('status') == 'ok'
                trial['object_dimensions_m'] = observation_data.get('dimensions_m')
                trial['failure_stage'] = 'segmentation' if not trial['segmentation_success'] else None
                run_logged(
                    ['ros2', 'service', 'call', '/pause_physics',
                     'std_srvs/srv/Empty', '{}'],
                    trial_dir / 'pause.log', base_env, 10,
                )

                candidates, first_metadata, roi_results = [], None, []
                trial['failure_stage'] = 'inference'
                for padding in roi_paddings:
                    roi_dir = trial_dir / f'roi_{padding:03d}'
                    roi_dir.mkdir()
                    run_logged(
                        [str(GRASPNET_PYTHON), 'scripts/graspnet_offline_demo.py',
                         '--input', str(frame), '--input-kind', 'gazebo',
                         '--roi-from-red-mask', str(padding),
                         '--roi-depth-window', '0.08',
                         '--checkpoint', str(CHECKPOINT),
                         '--output', str(roi_dir)],
                        roi_dir / 'inference.log', base_env, 90,
                    )
                    grasp_file = roi_dir / 'grasps_all.npy'
                    if not grasp_file.is_file():
                        raise RuntimeError(f'Missing inference output for ROI {padding}')
                    prepared = roi_dir / 'candidates.json'
                    run_logged(
                        [str(GRASPNET_PYTHON), 'scripts/prepare_graspnet_sim_candidates.py',
                         '--frame', str(frame),
                         '--metadata', str(trial_dir / 'frame.json'),
                         '--grasps', str(grasp_file),
                         '--out', str(prepared)],
                        roi_dir / 'prepare.log', base_env, 15,
                    )
                    prepared_data = json.loads(prepared.read_text())
                    roi_results.append({
                        'padding_px': padding,
                        'model_predictions': prepared_data['model_predictions'],
                        'accepted_candidates': len(prepared_data['candidates']),
                    })
                    if first_metadata is None:
                        first_metadata = prepared_data
                    for candidate in prepared_data['candidates']:
                        candidate = dict(candidate)
                        candidate['source_roi_padding_px'] = padding
                        candidates.append(candidate)
                trial['roi_results'] = roi_results
                trial['accepted_candidates'] = len(candidates)
                trial['candidate_count'] = len(candidates)
                trial['failure_stage'] = 'candidate_generation'
                if not candidates:
                    raise RuntimeError('No GraspNet candidates met opening and target filters')
                if first_metadata is None:
                    raise RuntimeError('No candidate metadata')
                candidates.sort(key=lambda item: (-item['score'], item['approach_distance_m']))
                for candidate_id, candidate in enumerate(candidates):
                    candidate['id'] = candidate_id
                merged = {
                    'source': 'graspnet_sim_rgbd',
                    'frame': 'base_link',
                    'sample_stamp': first_metadata['sample_stamp'],
                    'sample_joint_positions_rad': first_metadata['sample_joint_positions_rad'],
                    'T_base_camera': first_metadata['T_base_camera'],
                    'roi_sources': roi_results,
                    'candidates': candidates,
                }
                merged_path = trial_dir / 'candidates.json'
                write_json(merged_path, merged)
                trial['failure_stage'] = 'candidate_validation'
                validated_path = trial_dir / 'candidates_validated.json'
                run_logged(
                    [sys.executable, 'scripts/validate_grasp_candidates.py',
                     '--candidates', str(merged_path),
                     '--observation', str(trial_dir / 'frame_observation.json'),
                     '--out', str(validated_path)],
                    trial_dir / 'validate.log', base_env, 15,
                )
                validated_data = json.loads(validated_path.read_text())
                trial['revalidated_candidate_count'] = sum(
                    1 for c in validated_data['candidates'] if not c.get('rejection_reason'))
                trial['candidate_rejection_summary'] = validated_data.get('rejection_summary', {})
                trial['failure_stage'] = 'symmetry_augmentation'
                execution_path = trial_dir / 'symmetry_candidates.json'
                run_logged(
                    [sys.executable, 'scripts/augment_cylinder_grasp_symmetry.py',
                     '--candidates', str(validated_path),
                     '--observation', str(trial_dir / 'frame_observation.json'),
                     '--out', str(execution_path),
                     '--samples', str(args.symmetry_samples),
                     '--top-models', str(args.top_models),
                     *([] if args.observation_mode == 'generic' else
                       ['--object-shape', args.observation_mode])],
                    trial_dir / 'symmetry.log', base_env, 15,
                )
                selected_data = json.loads(execution_path.read_text())
                trial['symmetry_candidates'] = len(selected_data['candidates'])

                run_logged(
                    ['ros2', 'service', 'call', '/unpause_physics',
                     'std_srvs/srv/Empty', '{}'],
                    trial_dir / 'unpause.log', base_env, 40,
                )
                trial['failure_stage'] = 'execution'
                result_file = trial_dir / 'grasp_result.json'
                run_logged(
                    [sys.executable, 'sim/gazebo/grasp/run_grasp.py',
                     '--vision-observation', str(trial_dir / 'frame_observation.json'),
                     '--graspnet-candidates', str(execution_path),
                     *(['--geometry-fallback'] if args.geometry_fallback else []),
                     *(['--geometry-candidates'] if args.geometry_candidates else []),
                     '--grip-seconds', '8', '--hold-seconds', '5',
                     '--approach-scaling', '.03', '--lift-scaling', '.02',
                     '--out', str(result_file)],
                    trial_dir / 'action.log', base_env, 180, acceptable=(0, 1),
                )
                if not result_file.is_file():
                    raise RuntimeError('Grasp runner produced no result file')
                result = json.loads(result_file.read_text())
                selected = result.get('selected_candidate') or {}
                trial.update(
                    status='success' if result.get('execution_complete') else 'grasp_failed',
                    success=bool(result.get('success')),
                    execution_complete=bool(result.get('execution_complete')),
                    reason=result.get('reason'),
                    # run_grasp sets a real failure stage, or None on success;
                    # successful trials must not carry a failure stage.
                    failure_stage=result.get('failure_stage'),
                    failure_reason=result.get('reason'),
                    stages=result.get('stages', []),
                    stage_timings_s=result.get('stage_timings_s'),
                    vision_error_m=result.get('vision_error_m'),
                    object_lift_m=result.get('object_lift_m'),
                    max_object_lift_m=result.get('object_lift_m'),
                    hold_check=result.get('hold_stability') or result.get('cylinder_hold_check'),
                    hold_stability=result.get('hold_stability'),
                    hold_check_mode=result.get('hold_check_mode'),
                    grasp_state=result.get('grasp_state'),
                    grasp_state_details=result.get('grasp_state_details'),
                    commanded_close_width_m=result.get('commanded_close_width_m'),
                    motion_feedback_settled=result.get('lift_feedback_settled'),
                    feedback_failure=result.get('feedback_failure'),
                    selected_candidate=result.get('selected_candidate'),
                    selected_candidate_source=selected.get('source'),
                    selected_candidate_method=selected.get('method'),
                    candidate_attempt_count=len(result.get('candidate_attempts', [])),
                    planning_success=result.get('selected_candidate') is not None,
                    physical_grasp_success=bool(result.get('physical_grasp_success')),
                    revalidation_rejections=result.get('revalidation_rejections'),
                    collision_geometry_success=(
                        result.get('collision_geometry') is not None
                        or result.get('object_shape') in ('cylinder', 'cube')),
                    collision_geometry=result.get('collision_geometry'),
                    object_dimensions_m=result.get('object_dimensions_m'),
                    object_pose=result.get('object_pose'),
                )
                if SCENE_METADATA.is_file():
                    scene_params = json.loads(SCENE_METADATA.read_text())
                    trial['object_dimensions_scene_m'] = scene_params.get('object_dims_m')
                    trial['object_shape_scene'] = scene_params.get('object_shape')
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired,
                    ValueError, KeyError) as exc:
                trial.update(status='pipeline_failed', reason=str(exc),
                             failure_reason=str(exc))
                if trial.get('failure_stage') in (None, 'setup'):
                    trial['failure_stage'] = 'pipeline'
            finally:
                if not (args.keep_open_last and index + 1 == trial_count):
                    stop_scene(scene)
                trials_done = summary['trial_results']
                summary['completed_trials'] = len(trials_done)
                summary['success_count'] = sum(item['success'] for item in trials_done)
                summary['execution_complete_count'] = sum(
                    bool(item.get('execution_complete')) for item in trials_done)
                summary['physical_grasp_count'] = sum(
                    bool(item.get('physical_grasp_success')) for item in trials_done)
                summary['feedback_settled_count'] = sum(
                    1 for item in trials_done if item.get('motion_feedback_settled'))
                summary['candidate_selection_count'] = sum(
                    'grasp_failed' in item['status'] or item['status'] == 'success'
                    for item in trials_done)
                summary['success_rate'] = summary['success_count'] / trial_count
                # Separate statistics per candidate method: network scores are
                # not promoted by geometric rescues and vice versa.
                method_stats = {}
                for item in trials_done:
                    if not item.get('planning_success'):
                        continue
                    method = item.get('selected_candidate_method') or 'network'
                    entry = method_stats.setdefault(method, {'planned': 0, 'physical': 0,
                                                             'execution_complete': 0})
                    entry['planned'] += 1
                    entry['physical'] += int(bool(item.get('physical_grasp_success')))
                    entry['execution_complete'] += int(bool(item.get('execution_complete')))
                summary['method_stats'] = method_stats
                write_json(summary_path, summary)
                print(f"Trial {index + 1}: {trial['status']}", flush=True)
        summary['status'] = 'completed'
        if args.keep_open_last:
            write_json(summary_path, summary)
            print('KEEPING_FINAL_GAZEBO_SCENE_OPEN', flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                stop_scene(scene)
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
        raise
    finally:
        write_json(summary_path, summary)
        print(f'Saved {summary_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
