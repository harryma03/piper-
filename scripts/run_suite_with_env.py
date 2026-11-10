#!/usr/bin/env python3
"""Run a scene suite through a batch script that lacks --scenes-json support.

One subprocess per scene: the batch runner is invoked with a single explicit
--xy (plus --object-shape) while the scene's dims/yaw/occluder/view pass
through their PIPER_SIM_* environment variables. Used to drive the baseline
snapshot code over the same fixed suite as the refactored pipeline, so the
before/after comparison covers identical scenes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--batch-script', type=Path, required=True,
                        help='Path to run_gazebo_graspnet_batch.py (snapshot or current)')
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--extra-args', default='',
                        help='Extra batch flags, e.g. "--top-models 4 --symmetry-samples 8"')
    parser.add_argument('--limit', type=int, default=0, help='Run only the first N scenes')
    parser.add_argument('--supported-shapes', default='cylinder,cube,sphere,box,bar,mesh',
                        help='Shapes the target batch script accepts; others are skipped and recorded')
    parser.add_argument('--dims-fallback', default='',
                        help='--object-dims value when the scene spec has no dims list')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Resume mode: skip scenes whose out_dir already has a trial_001 result')
    parser.add_argument('--skip-unsupported-dims', action='store_true',
                        help='Skip scenes needing occluder/view variants the target cannot express')
    args = parser.parse_args()
    payload = json.loads(args.suite.read_text())
    scenes = payload['scenes'] if isinstance(payload, dict) else payload
    if args.limit:
        scenes = scenes[:args.limit]
    args.out_root.mkdir(parents=True, exist_ok=True)
    extra = args.extra_args.split()
    failures = []
    skipped = []
    for index, spec in enumerate(scenes):
        out_dir = args.out_root / f'trial_{index + 1:03d}'
        if args.skip_existing and (out_dir / 'trial_001' / 'grasp_result.json').is_file():
            print(f'[{index + 1}/{len(scenes)}] resume: {out_dir} already done', flush=True)
            continue
        supported = args.supported_shapes.split(',')
        baseline_cannot_express = (spec.get('occluder') or int(spec.get('view', 0) or 0) != 0)
        if spec['shape'] not in supported or (args.skip_unsupported_dims
                                              and baseline_cannot_express):
            skipped.append({'scene': index + 1, 'shape': spec['shape'],
                            'reason': ('baseline pipeline cannot parameterize this shape'
                                       if spec['shape'] not in supported else
                                       'baseline cannot express occluder/view variants')})
            (out_dir).mkdir(parents=True, exist_ok=True)
            (out_dir / 'skipped.json').write_text(json.dumps(
                {'shape': spec['shape'], 'reason': 'baseline scene parameters unsupported'},
                indent=1) + '\n')
            print(f'[{index + 1}/{len(scenes)}] SKIPPED {spec["shape"]} (baseline unsupported)',
                  flush=True)
            continue
        env = dict(os.environ)
        env['PIPER_SIM_OBJECT_DIMS'] = ','.join(f'{float(v):.4f}' for v in spec.get('dims', [])) \
            if isinstance(spec.get('dims'), list) else ''
        env['PIPER_SIM_OBJECT_YAW_DEG'] = f"{float(spec.get('yaw', 0)):.4f}"
        env['PIPER_SIM_OCCLUDER'] = '1' if spec.get('occluder') else '0'
        env['PIPER_SIM_VIEW'] = str(int(spec.get('view', 0)))
        batch_abs = Path(args.batch_script).resolve()
        command = [sys.executable, str(batch_abs),
                   '--object-shape', spec['shape'],
                   '--xy', f"{spec['xy'][0]},{spec['xy'][1]}",
                   '--object-dims',
                   (','.join(f'{float(v):.4f}' for v in spec.get('dims'))
                    if isinstance(spec.get('dims'), list) else args.dims_fallback),
                   '--object-yaw', f"{float(spec.get('yaw', 0)):.4f}",
                   '--out', str(out_dir.resolve())] + extra
        print(f'[{index + 1}/{len(scenes)}] {spec["shape"]} dims={spec.get("dims")} '
              f'xy={spec["xy"]} -> {out_dir}', flush=True)
        result = subprocess.run(command, env=env, cwd=str(batch_abs.parents[1]))
        if result.returncode not in (0, 1):
            failures.append({'scene': index + 1, 'returncode': result.returncode})
        # The batch writes trial_001 inside out_dir; flatten for consistency.
        inner = out_dir / 'trial_001'
        if inner.is_dir():
            summary = out_dir / 'summary.json'
            if not summary.exists():
                try:
                    (out_dir / 'summary.json').write_text((inner / '..' / 'summary.json').resolve().read_text())
                except OSError:
                    pass
    print(f'completed {len(scenes)} scenes; infra failures: {failures}; '
          f'skipped (baseline unsupported): {len(skipped)}')
    (args.out_root / 'skipped_scenes.json').write_text(
        json.dumps(skipped, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
