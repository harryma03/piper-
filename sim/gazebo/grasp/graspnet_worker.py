#!/usr/bin/env python3
"""Persistent GraspNet inference worker (run inside .graspnet-venv).

Speaks a line-based JSON protocol over stdin/stdout so the ROS-side
perception service (system python, no torch) can keep the model resident
on the GPU instead of paying process start-up per request:

    request : {"id": int, "points": "/abs/path.npy", "num_point": 20000,
               "seed": int, "min_depth": 0.08, "max_depth": 2.5}
    response: {"id": int, "ok": true, "predictions": "/abs/path.npy",
               "raw_count": int, "time_ms": float}
    response: {"id": int, "ok": false, "error": "..."}

Run with the graspnet venv python. Exits cleanly on EOF.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / 'third_party/graspnet-baseline'


def load_model(checkpoint_path):
    for path in (BASELINE / 'models', BASELINE / 'dataset', BASELINE / 'utils', BASELINE):
        sys.path.insert(0, str(path))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for the persistent worker')
    from graspnet import GraspNet
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04],
                   is_training=False).cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


def infer(net, points_path, num_point, seed, min_depth, max_depth):
    import numpy as np
    import torch
    from graspnet import pred_decode
    points_all = np.load(points_path, allow_pickle=False)
    valid = np.isfinite(points_all).all(axis=-1) & \
        (points_all[:, 2] > min_depth) & (points_all[:, 2] < max_depth)
    points = points_all[valid]
    if len(points) < 100:
        raise ValueError(f'only {len(points)} valid input points')
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), num_point, replace=len(points) < num_point)
    with torch.no_grad():
        data = {'point_clouds': torch.from_numpy(points[idx][None].astype(np.float32)).cuda()}
        predictions = pred_decode(net(data))[0].detach().cpu().numpy()
    return predictions, points


def main():
    import numpy as np  # request loop saves prediction arrays
    request = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    checkpoint = Path(request['checkpoint']) if request and 'checkpoint' in request \
        else ROOT / 'third_party/graspnet-baseline/checkpoints/checkpoint-rs.tar'
    if 'axis_z = torch.cross(axis_x, axis_y, dim=-1)' not in \
            (BASELINE / 'utils/loss_utils.py').read_text():
        print(json.dumps({'id': -1, 'ok': False,
                          'error': 'apply patches/graspnet_baseline_cross.patch first'}),
              flush=True)
        return
    net = load_model(checkpoint)
    print(json.dumps({'id': -1, 'ok': True, 'event': 'ready'}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        started = time.monotonic()
        try:
            req = json.loads(line)
            predictions, _ = infer(net, req['points'],
                                   int(req.get('num_point', 20000)),
                                   int(req.get('seed', 42)),
                                   float(req.get('min_depth', 0.08)),
                                   float(req.get('max_depth', 2.5)))
            out_path = req.get('output')
            if not out_path:
                out_path = str(Path(req['points']).parent / f"pred_{req['id']}.npy")
            np.save(out_path, predictions.astype(np.float32))
            print(json.dumps({'id': req['id'], 'ok': True, 'predictions': out_path,
                              'raw_count': int(len(predictions)),
                              'time_ms': round((time.monotonic() - started) * 1000, 1)}),
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - worker must survive bad requests
            print(json.dumps({'id': req.get('id', -1) if isinstance(req, dict) else -1,
                              'ok': False, 'error': str(exc),
                              'time_ms': round((time.monotonic() - started) * 1000, 1)}),
                  flush=True)


if __name__ == '__main__':
    main()
