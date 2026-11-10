#!/usr/bin/env python3
"""Run GraspNet Baseline on a saved RGB-D observation; never command a robot.

For C1W recordings, consume the saved SDK XYZ cloud rather than applying a
pinhole model to the camera's fisheye depth image. Predicted frames are the
GraspNet gripper convention in the camera frame, not Piper TCP poses.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_recording(path, roi_padding, depth_window=None):
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data['xyz'], dtype=np.float32)
        # diagnose_cylinder_geometry.py saves OpenCV BGR.
        color = np.asarray(data['color'], dtype=np.uint8)[..., ::-1].copy()
        if xyz.ndim != 3 or xyz.shape[2] != 3 or color.shape != xyz.shape:
            raise ValueError('Expected organized xyz/color arrays of shape H x W x 3')
        roi = np.ones(xyz.shape[:2], dtype=bool)
        target_points = None
        center = None
        if roi_padding is not None:
            if 'mask' not in data:
                raise ValueError('The recording has no observed target mask for ROI')
            target = np.asarray(data['mask'], dtype=bool)
            if target.shape != roi.shape or not target.any():
                raise ValueError('The recorded target mask is empty or has invalid size')
            y, x = np.nonzero(target)
            roi[:] = False
            roi[max(0, y.min()-roi_padding):min(roi.shape[0], y.max()+roi_padding+1),
                max(0, x.min()-roi_padding):min(roi.shape[1], x.max()+roi_padding+1)] = True
            target_points = xyz[target & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)]
            if len(target_points) == 0:
                raise ValueError('No valid point inside the observed target mask')
            center = np.median(target_points, axis=0)
            if depth_window is not None:
                roi &= np.abs(xyz[..., 2]-center[2]) <= depth_window
    return xyz, color, roi, 'C1W SDK organized XYZ; observed RGB-D, BGR converted to RGB', target_points, center


def load_example(root):
    from PIL import Image
    from scipy.io import loadmat
    folder = root / 'doc/example_data'
    color = np.asarray(Image.open(folder / 'color.png').convert('RGB'), dtype=np.uint8)
    depth = np.asarray(Image.open(folder / 'depth.png'), dtype=np.float32)
    roi = np.asarray(Image.open(folder / 'workspace_mask.png')) > 0
    meta = loadmat(folder / 'meta.mat')
    k = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
    scale = float(np.asarray(meta['factor_depth']).reshape(-1)[0])
    if depth.ndim != 2 or color.shape[:2] != depth.shape or roi.shape != depth.shape or scale <= 0:
        raise ValueError('Example image shapes or depth factor are invalid')
    v, u = np.indices(depth.shape, dtype=np.float32)
    z = depth / scale
    xyz = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=-1)
    return xyz.astype(np.float32), color, roi, 'official GraspNet example; pinhole depth and workspace mask', None, None


def select_diverse(pred, limit=10, translation_m=0.02, angle_deg=20.):
    """Visualization-only spatial/angular thinning; no collision or IK checking."""
    selected = []
    for g in pred[np.argsort(pred[:, 0])[::-1]]:
        R, t = g[4:13].reshape(3, 3), g[13:16]
        too_similar = False
        for kept in selected:
            if np.linalg.norm(t - kept[13:16]) >= translation_m:
                continue
            relative = R.T @ kept[4:13].reshape(3, 3)
            angle = np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))
            if angle < angle_deg:
                too_similar = True
                break
        if not too_similar:
            selected.append(g)
            if len(selected) == limit:
                break
    return np.asarray(selected, dtype=np.float32).reshape(-1, 17)


def render(path, points, colors, grasps, seed, local_center=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    if local_center is None and len(grasps):
        local_center = grasps[0, 13:16]
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    for index, (elev, azim) in enumerate(((20, -65), (70, -90)), 1):
        ax = fig.add_subplot(1, 2, index, projection='3d')
        if index == 2 and local_center is not None:
            in_view = (np.abs(points-local_center) <= .15).all(axis=1)
            plot_points, plot_colors = points[in_view], colors[in_view]
        else:
            plot_points, plot_colors = points, colors
        if len(plot_points) > 18000:
            ind = rng.choice(len(plot_points), 18000, replace=False)
            plot_points, plot_colors = plot_points[ind], plot_colors[ind]
        ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], c=plot_colors/255., s=.5, alpha=.55, rasterized=True)
        for i, g in enumerate(grasps):
            width, depth = float(g[1]), float(g[3])
            R, t = g[4:13].reshape(3, 3), g[13:16]
            # GraspNet uses local X as approach and local Y for finger opening.
            vertices = np.array([[depth-.06, -width/2, 0], [depth, -width/2, 0],
                                 [depth-.06, -width/2, 0], [depth-.06, width/2, 0],
                                 [depth-.06, width/2, 0], [depth, width/2, 0]]) @ R.T + t
            for j in (0, 2, 4):
                ax.plot(*vertices[j:j+2].T, color=plt.cm.tab10(i % 10), linewidth=1.6)
        ax.set_xlabel('camera X (m)')
        ax.set_ylabel('camera Y (m)')
        ax.set_zlabel('camera Z (m)')
        ax.set_title('Full scene' if index == 1 else 'Local view around observed target')
        ax.view_init(elev=elev, azim=azim)
        if index == 2 and local_center is not None:
            ax.set_xlim(local_center[0]-.15, local_center[0]+.15)
            ax.set_ylim(local_center[1]-.15, local_center[1]+.15)
            ax.set_zlim(local_center[2]-.15, local_center[2]+.15)
            ax.set_box_aspect([1, 1, 1])
        else:
            ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), .01))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='Recorded frame.npz with organized XYZ and BGR color')
    source.add_argument('--example', action='store_true', help='Official RGB-D example')
    p.add_argument('--input-kind', choices=('c1w', 'gazebo'), default='c1w',
                   help='Source label for recorded point clouds; gazebo is the ideal simulated sensor')
    p.add_argument('--checkpoint', type=Path, required=True, help='GraspNet model checkpoint (verify provenance if obtained from a third party)')
    p.add_argument('--output', type=Path, required=True, help='Output directory')
    p.add_argument('--baseline-root', type=Path, default=ROOT / 'third_party/graspnet-baseline')
    p.add_argument('--roi-from-mask', '--roi-from-red-mask', dest='roi_from_red_mask', type=int, metavar='PADDING', help='Limit model input to bbox around recorded target mask plus this many pixels; full scene remains for plotting')
    p.add_argument('--roi-depth-window', type=float, metavar='METERS', help='Also limit input depth to this distance either side of the observed target-mask median')
    p.add_argument('--min-depth', type=float, default=.08)
    p.add_argument('--max-depth', type=float, default=2.5)
    p.add_argument('--num-point', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.min_depth < 0 or args.max_depth <= args.min_depth or args.num_point < 1:
        p.error('invalid depth range or point count')
    if args.roi_from_red_mask is not None and (args.example or args.roi_from_red_mask < 0):
        p.error('--roi-from-red-mask requires a recording and a non-negative padding')
    if args.roi_depth_window is not None and (args.roi_from_red_mask is None or args.roi_depth_window <= 0):
        p.error('--roi-depth-window requires --roi-from-red-mask and a positive window')
    baseline = args.baseline_root.resolve()
    if not (baseline / 'models/graspnet.py').is_file():
        p.error(f'Cannot find GraspNet source at {baseline}')
    if 'axis_z = torch.cross(axis_x, axis_y, dim=-1)' not in (baseline / 'utils/loss_utils.py').read_text():
        p.error('Apply patches/graspnet_baseline_cross.patch in the baseline checkout before inference')
    if not args.checkpoint.is_file():
        p.error(f'Checkpoint not found: {args.checkpoint}')
    xyz, color, roi, source_desc, target_points, target_center = (load_example(baseline) if args.example else
                                    load_recording(args.input, args.roi_from_red_mask, args.roi_depth_window))
    if not args.example and args.input_kind == 'gazebo':
        source_desc = 'Gazebo simulated RGB-D, organized XYZ; BGR converted to RGB'
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > args.min_depth) & (xyz[..., 2] < args.max_depth)
    scene_points, scene_colors = xyz[valid], color[valid]
    points, colors = xyz[valid & roi], color[valid & roi]
    if len(points) < 100:
        raise ValueError(f'Only {len(points)} valid input points; check ROI and depth range')
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(points), args.num_point, replace=len(points)<args.num_point)
    for path in (baseline / 'models', baseline / 'dataset', baseline / 'utils', baseline):
        sys.path.insert(0, str(path))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GraspNet baseline CUDA point operators require a CUDA-capable PyTorch environment')
    from graspnet import GraspNet, pred_decode
    from loss_utils import batch_viewpoint_params_to_matrix
    # Three candidates exercise the original torch.cross dimension ambiguity.
    test_rotations = batch_viewpoint_params_to_matrix(torch.eye(3), torch.zeros(3))
    identity = torch.eye(3).expand(3, -1, -1)
    if not torch.allclose(test_rotations.transpose(1, 2) @ test_rotations, identity, atol=1e-5):
        raise RuntimeError('GraspNet rotation decoder failed the three-candidate orthogonality check; apply patches/graspnet_baseline_cross.patch')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False).cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    with torch.no_grad():
        data = {'point_clouds': torch.from_numpy(points[idx][None].astype(np.float32)).cuda()}
        predictions = pred_decode(net(data))[0].detach().cpu().numpy()
    if predictions.ndim != 2 or predictions.shape[1] != 17:
        raise ValueError(f'Unexpected prediction shape {predictions.shape}')
    raw_predictions = predictions.copy()
    raw_count = len(predictions)
    # Only visualize finite, physically meaningful decoded gripper geometry.
    rotations = predictions[:, 4:13].reshape(-1, 3, 3)
    orthogonality_error = np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)), axis=(1, 2))
    valid_prediction = (np.isfinite(predictions).all(axis=1) & (predictions[:, 1] > 0) &
                        (predictions[:, 1] <= .1) & (predictions[:, 2] > 0) &
                        (predictions[:, 3] > 0) & (orthogonality_error < 1e-3) &
                        (np.abs(np.linalg.det(rotations) - 1) < 1e-3))
    predictions = predictions[valid_prediction]
    best = select_diverse(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'grasps_all.npy', raw_predictions)
    np.save(args.output / 'grasps_valid.npy', predictions)
    np.save(args.output / 'grasps_top10.npy', best)
    summary = {'source': source_desc, 'input_file': str(args.input) if args.input else 'official_example',
               'checkpoint': str(args.checkpoint), 'frame': 'camera optical frame',
               'scene_points': int(len(scene_points)), 'input_points_before_sampling': int(len(points)),
               'sampled_points': int(args.num_point), 'raw_predictions_count': int(raw_count),
               'predictions_count_after_geometry_check': int(len(predictions)),
               'selected_count': int(len(best)), 'roi_padding_pixels': args.roi_from_red_mask,
               'roi_depth_window_m': args.roi_depth_window,
               'observed_target_surface_median_m': target_center.tolist() if target_center is not None else None,
               'selection': 'score sorted, 2cm/20deg visual de-duplication',
               'decoder_fix': 'requires patches/graspnet_baseline_cross.patch (torch.cross dim=-1)',
               'collision_check': False, 'robot_reachability_check': False,
               'warning': 'Research visualization only; no TCP conversion, collision checks, reachability or actuator commands.',
               'grasps': [{'score': float(g[0]), 'width_m': float(g[1]), 'height_m': float(g[2]),
                            'depth_m': float(g[3]), 'rotation_matrix': g[4:13].reshape(3,3).tolist(),
                            'translation_m': g[13:16].tolist(),
                            'nearest_observed_target_surface_m': float(np.linalg.norm(target_points-g[13:16], axis=1).min()) if target_points is not None else None} for g in best]}
    (args.output / 'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    render(args.output / 'grasps.png', scene_points, scene_colors, best, args.seed, target_center)
    print(json.dumps({k: v for k, v in summary.items() if k != 'grasps'}, indent=2))


if __name__ == '__main__':
    main()
