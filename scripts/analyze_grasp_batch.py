#!/usr/bin/env python3
"""Summarize a grasp batch: per-group endpoints, failure stages, methods.

Reads one or more batch summary.json files and prints a table using the
pre-registered endpoint definitions. Spawn truth is used only for the
perception-error column, never to re-judge outcomes.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_trials(path, suite=None):
    """Load trials from a batch summary or a per-scene wrapper directory.

    Handles both layouts (and a mix): a batch summary file, standard
    ``trial_XXX/grasp_result.json`` trees, and wrapper runs that nest each
    scene's batch output under ``trial_XXX/trial_001/``.
    """
    path = Path(path)
    if path.is_dir():
        rows = []
        for index, trial_dir in enumerate(sorted(path.glob('trial_*'))):
            candidates = [trial_dir / 'grasp_result.json',
                          trial_dir / 'trial_001' / 'grasp_result.json']
            result_path = next((c for c in candidates if c.is_file()), None)
            skipped = trial_dir / 'skipped.json'
            if result_path is None:
                if skipped.is_file():
                    rows.append({'index': index + 1, 'status': 'skipped_baseline_unsupported',
                                 'group': '', 'segmentation': False, 'planning': False,
                                 'physical': False, 'settled': None, 'complete': False,
                                 'failure_stage': 'baseline_unsupported', 'method': None,
                                 'lift': None, 'grasp_state': None, 'spawn': None,
                                 'observed_center': None, 'shape': None, 'dims': None})
                else:
                    rows.append({'index': index + 1, 'status': 'missing', 'group': '',
                                 'segmentation': False, 'planning': False, 'physical': False,
                                 'settled': None, 'complete': False, 'failure_stage': 'missing',
                                 'method': None, 'lift': None, 'grasp_state': None,
                                 'spawn': None, 'observed_center': None, 'shape': None,
                                 'dims': None})
                continue
            result = json.loads(result_path.read_text())
            observation = None
            obs_path = next((c for c in [trial_dir / 'frame_observation.json',
                                         trial_dir / 'trial_001' / 'frame_observation.json']
                             if c.is_file()), None)
            if obs_path is not None:
                observation = json.loads(obs_path.read_text())
            rows.append(_trial_row({'status': 'success' if result.get('execution_complete')
                                    else 'grasp_failed',
                                    'planning_success': result.get('selected_candidate') is not None,
                                    'physical_grasp_success': bool(result.get('physical_grasp_success')),
                                    'motion_feedback_settled': result.get('lift_feedback_settled'),
                                    'execution_complete': bool(result.get('execution_complete')),
                                    'failure_stage': result.get('failure_stage'),
                                    'selected_candidate_method': (result.get('selected_candidate') or {}).get('method'),
                                    'max_object_lift_m': result.get('object_lift_m'),
                                    'grasp_state': result.get('grasp_state'),
                                    'observation': observation}, index))
        return {'layout': 'tree'}, rows
    summary = json.loads(path.read_text())
    rows = []
    for index, trial in enumerate(summary.get('trial_results') or []):
        rows.append(_trial_row(trial, index))
    return summary, rows


def _trial_row(trial, index):
    return {
        'index': index + 1,
        'group': trial.get('group') or '',
        'shape': trial.get('object_shape'),
        'dims': trial.get('object_dims_scene_m') or trial.get('object_dims_m'),
        'status': trial.get('status'),
        'segmentation': bool(trial.get('segmentation_success')),
        'planning': bool(trial.get('planning_success')),
        'physical': bool(trial.get('physical_grasp_success')),
        'settled': trial.get('motion_feedback_settled'),
        'complete': bool(trial.get('execution_complete')),
        'failure_stage': trial.get('failure_stage'),
        'method': trial.get('selected_candidate_method') or (
            'network' if trial.get('planning_success') else None),
        'lift': trial.get('max_object_lift_m'),
        'grasp_state': trial.get('grasp_state'),
        'spawn': trial.get('spawn_xy_m'),
        'observed_center': (trial.get('observation') or {}).get('obb_center_base_m')
        if isinstance(trial.get('observation'), dict) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='+', type=Path)
    parser.add_argument('--suite', type=Path, default=None,
                        help='Suite JSON to join group labels by trial order')
    args = parser.parse_args()
    suite = None
    if args.suite:
        payload = json.loads(args.suite.read_text())
        suite = payload['scenes'] if isinstance(payload, dict) else payload
    for path in args.summaries:
        summary, rows = load_trials(path, suite)
        if suite:
            for row in rows:
                if row['index'] <= len(suite):
                    row['group'] = suite[row['index'] - 1].get('group', '')
        valid = [r for r in rows if r['status'] != 'pipeline_failed'
                 or r['failure_stage'] not in ('setup', 'pipeline')]
        infra = [r for r in rows if r not in valid]
        print(f"\n=== {path} ===")
        print(f"trials: {len(rows)} | infra/pipeline failures excluded: {len(infra)} "
              f"| valid denominator: {len(valid)}")
        if not valid:
            continue
        def rate(key):
            return sum(1 for r in valid if r[key]) / len(valid)
        print(f"segmentation:      {sum(r['segmentation'] for r in valid)}/{len(valid)}")
        print(f"planning:          {sum(r['planning'] for r in valid)}/{len(valid)}")
        print(f"physical_grasp:    {sum(r['physical'] for r in valid)}/{len(valid)}")
        settled_known = [r for r in valid if r['settled'] is not None]
        print(f"feedback_settled:  {sum(1 for r in settled_known if r['settled'])}/{len(settled_known)} "
              f"(of {len(settled_known)} executed)")
        print(f"execution_complete:{sum(r['complete'] for r in valid)}/{len(valid)}")
        print(f"failure stages: {dict(Counter(str(r['failure_stage']) for r in valid if not r['complete']))}")
        print(f"grasp states:   {dict(Counter(str(r['grasp_state']) for r in valid))}")
        methods = Counter(r['method'] for r in valid if r['planning'])
        print(f"selected methods: {dict(methods)}")
        by_group = {}
        for r in valid:
            by_group.setdefault(r['group'], []).append(r)
        for group, group_rows in sorted(by_group.items()):
            physical = sum(r['physical'] for r in group_rows)
            complete = sum(r['complete'] for r in group_rows)
            print(f"  {group:18s} n={len(group_rows):2d} physical={physical} complete={complete}")
        # Perception error vs spawn truth (XY only; truth for evaluation only).
        errors = []
        for r in valid:
            if r['spawn'] and r['observed_center']:
                err = np.linalg.norm(np.array(r['observed_center'][:2]) - np.array(r['spawn']))
                errors.append(err)
        if errors:
            print(f"perception xy error: max {max(errors)*1e3:.1f} mm, "
                  f"mean {float(np.mean(errors))*1e3:.1f} mm, <5mm {sum(e<0.005 for e in errors)}/{len(errors)}")


if __name__ == '__main__':
    main()
