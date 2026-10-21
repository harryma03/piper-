"""Audit a stored planning preview offline. Never commands hardware or grants execution."""
import argparse
import json
import math
from pathlib import Path
import time


def audit_preview(report, now=None, max_age=1.0, max_step=0.15):
    now = time.time() if now is None else now
    errors = []
    if report.get('status') != 'preview_path_found':
        errors.append('No successful preview path')
    stamp = report.get('observation', {}).get('stamp', {})
    try:
        age = now - (stamp['sec'] + stamp['nanosec'] * 1e-9)
        if not math.isfinite(age) or not 0 <= age <= max_age:
            errors.append('Observation is stale or has a future timestamp; reobserve and replan')
    except (KeyError, TypeError):
        age = None
        errors.append('Missing observation timestamp')
    initial = report.get('initial_real_joint_state', {})
    names, positions = initial.get('names', []), initial.get('positions', [])
    if (len(names) != len(positions) or len(set(names)) != len(names) or
            not all(isinstance(v, (int, float)) and math.isfinite(v) for v in positions)):
        errors.append('Invalid initial joint state')
    previous = dict(zip(names, positions))
    arm = {f'joint{i}' for i in range(1, 7)}
    if not arm.issubset(previous):
        errors.append('Missing initial arm joints')
    segments = report.get('trajectories', [])
    if len(segments) != 2:
        errors.append('Expected pregrasp and approach trajectories')
    peak = 0.0
    for index, segment in enumerate(segments):
        joints, points = segment.get('joint_names', []), segment.get('points', [])
        if len(joints) != len(set(joints)) or not arm.issubset(joints) or not points:
            errors.append(f'Segment {index}: missing or duplicate joints / empty path')
            continue
        last_time = -1.0
        for point in points:
            values, seconds = point.get('positions', []), point.get('time_from_start_s')
            if (len(values) != len(joints) or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)):
                errors.append(f'Segment {index}: invalid positions')
                break
            if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or
                    seconds < 0 or seconds < last_time):
                errors.append(f'Segment {index}: invalid trajectory timing')
                break
            last_time = seconds
            current = dict(zip(joints, values))
            for name in arm:
                if name in previous:
                    step = abs(current[name] - previous[name])
                    peak = max(peak, step)
                    if step > max_step:
                        errors.append(f'Segment {index}: {name} step exceeds {max_step} rad')
            previous = current
    return {
        'execution_ready': False,
        'record_checks_pass': not errors,
        'observation_age_s': age,
        'maximum_arm_sample_step_rad': peak,
        'record_issues': list(dict.fromkeys(errors)),
        'required_before_real_grasp': [
            'Measure/validate gripper contact TCP; model fingertip is only a reference',
            'Verify physical jaw opening against feedback and validate contact/force detection',
            'Model full cylinder including white ends and surrounding obstacles',
            'Plan closure, attached-object lift and retreat; validate grasp success and abort behavior',
            'Acquire fresh synchronized perception and real joint feedback before each motion',
        ],
        'scope': 'Offline record checks only; no live scene, reachability or actuator validation',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preview', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit_preview(json.loads(args.preview.read_text()))
    output = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end='')
    return 0 if result['record_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
