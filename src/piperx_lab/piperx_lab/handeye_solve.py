"""Solve Piper wrist-to-camera transform from saved observations."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from piperx_lab.handeye import solve_eye_in_hand
from piperx_lab.workspace import workspace_root


def main(argv=None):
    parser = argparse.ArgumentParser(description='Solve eye-in-hand calibration')
    parser.add_argument('--samples', type=Path,
                        help='Sample directory; defaults to <workspace>/calibration/samples')
    parser.add_argument('--out', type=Path,
                        help='Result file; defaults to <workspace>/calibration/handeye_result.json')
    parser.add_argument('--min-samples', type=int, default=8)
    parser.add_argument('--mode', choices=('real', 'sim'), default='real',
                        help='Solve real samples by default; sim is for Gazebo validation')
    parser.add_argument('--camera-frame', default='camera_depth_frame',
                        help='Frame the samples must share; must match camera_target')
    args = parser.parse_args(argv)
    if args.samples is None or args.out is None:
        try:
            root = workspace_root()
        except FileNotFoundError as exc:
            parser.error(str(exc))
        args.samples = args.samples or root / 'calibration/samples'
        args.out = args.out or root / 'calibration/handeye_result.json'
    if args.min_samples < 5:
        parser.error('--min-samples must be at least 5')
    files = sorted(args.samples.glob('*.json'))
    samples = []
    used_files = []
    for path in files:
        item = json.loads(path.read_text(encoding='utf-8'))
        if item.get('mode') != args.mode or item.get('robot') != 'piper':
            continue
        samples.append(item)
        used_files.append(path)
    if len(samples) < args.min_samples:
        print(f'Need {args.min_samples} {args.mode} Piper samples; found {len(samples)}',
              file=sys.stderr)
        return 1
    frame_pairs = {(item.get('base_frame'), item.get('wrist_frame')) for item in samples}
    if (len(frame_pairs) != 1 or next(iter(frame_pairs))[0] != 'base_link'
            or next(iter(frame_pairs))[1] != 'link6'):
        print(f'Inconsistent or unsupported TF frames: {frame_pairs}', file=sys.stderr)
        return 1
    try:
        result = solve_eye_in_hand(samples)
    except ValueError as exc:
        print(f'Calibration failed: {exc}', file=sys.stderr)
        return 1
    frames = {item['camera_frame'] for item in samples}
    if frames != {args.camera_frame}:
        print(f'Samples must all use {args.camera_frame}; found {frames}. '
              'Recapture with handeye_capture --camera-source depth.',
              file=sys.stderr)
        return 1
    cameras = {item.get('camera') for item in samples}
    if len(cameras) != 1:
        print(f'Mixed camera models in samples: {cameras}', file=sys.stderr)
        return 1
    result['camera_frame'] = args.camera_frame
    result['robot'] = 'piper'
    result['camera'] = next(iter(cameras))
    result['mode'] = args.mode
    result['base_frame'], result['wrist_frame'] = next(iter(frame_pairs))
    result['source_samples'] = [str(path) for path in used_files]
    result['mean_reprojection_rms_px'] = (
        sum(float(item['reprojection_rms_px']) for item in samples) / len(samples))
    result['quality_pass'] = (
        result['fixed_board_position_spread_max_m'] < 0.02 and
        result['fixed_board_rotation_spread_max_deg'] < 5.0 and
        result['mean_reprojection_rms_px'] < 2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print(f'Saved {args.out}; quality_pass={result["quality_pass"]}')
    translation = np.asarray(result['T_wrist_camera'], dtype=np.float64)[:3, 3]
    print(f'Wrist->camera offset: x={translation[0]:+.4f} y={translation[1]:+.4f} '
          f'z={translation[2]:+.4f} m, distance={np.linalg.norm(translation):.4f} m')
    print(f'Board consistency: max position spread '
          f'{result["fixed_board_position_spread_max_m"]:.4f} m, '
          f'max rotation spread {result["fixed_board_rotation_spread_max_deg"]:.2f} deg')
    return 0 if result['quality_pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
