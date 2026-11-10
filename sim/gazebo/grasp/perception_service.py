#!/usr/bin/env python3
"""Persistent perception service: continuous RGB-D + resident inference.

Runs on ROS domain 183 next to the simulation. Unlike capture_graspnet.py
(one capture per process) this node stays alive, keeps the latest
synchronised RGB-D bundle cached by the subscriber callbacks, and performs
the heavy work -- colour segmentation, generic observation, GraspNet
inference through the persistent worker, and TCP conversion -- on a worker
thread. ROS callbacks never block on compute.

Protocol (std_msgs/String, JSON):
    /piperx/perception/request  {"request_id": int, "view_id": str,
                                 "save_dir": "/abs/dir",
                                 "roi_paddings": [40, 60], "depth_window": 0.08}
    /piperx/perception/response {"request_id": int, "view_id": str,
                                 "ok": bool, "observation": {...},
                                 "candidates": [...], "timing_ms": {...},
                                 "frame_npz": path, "error": ...}

Truth isolation: no Gazebo state service is touched; everything the service
answers with derives from the camera stream, TF and joint states.
"""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.time import Time  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, JointState  # noqa: E402
from std_msgs.msg import Float32, String  # noqa: E402
from tf2_ros import Buffer, TransformListener, TransformException  # noqa: E402

from observe_cylinder import SimObservation  # noqa: E402
from piperx_lab.camera_target import decode_color, decode_depth  # noqa: E402
from piperx_lab.grasp_candidates import apply_revalidation  # noqa: E402
from piperx_lab.graspnet_adapter import convert_grasps  # noqa: E402
from piperx_lab.cylinder_grasp import model_tip_transform  # noqa: E402
from piperx_lab.model import installed_gripper_model_path  # noqa: E402
from piperx_lab.object_observation import (  # noqa: E402
    build_generic_observation, save_generic_observation)
from piperx_lab.rgbd_geometry import (  # noqa: E402
    aligned_frames_valid, color_region_points, depth_points_camera)
from piperx_lab.handeye import camera_model, transform_from_ros  # noqa: E402
from piperx_lab.depth_selection import select_depth_component  # noqa: E402


class PersistentGraspnet:
    """Line-protocol client for the resident inference worker process."""

    def __init__(self, checkpoint=None, timeout_s=90.0):
        worker = str(Path(__file__).resolve().parent / 'graspnet_worker.py')
        venv_python = ROOT / '.graspnet-venv/bin/python'
        if not venv_python.is_file():
            raise RuntimeError('graspnet venv missing')
        cmd = [str(venv_python), worker]
        if checkpoint:
            cmd.append(json.dumps({'checkpoint': str(checkpoint)}))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.timeout_s = timeout_s
        ready = json.loads(self.proc.stdout.readline())
        if not ready.get('ok'):
            raise RuntimeError(f"worker failed to start: {ready.get('error')}")
        self.request_counter = 0

    def infer(self, points_path, output_path, num_point=20000, seed=42):
        self.request_counter += 1
        request_id = self.request_counter
        self.proc.stdin.write(json.dumps({
            'id': request_id, 'points': str(points_path),
            'output': str(output_path), 'num_point': num_point, 'seed': seed}) + '\n')
        self.proc.stdin.flush()
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError('worker died')
            reply = json.loads(line)
            if reply.get('id') != request_id:
                continue  # stale reply for a timed-out earlier request
            if not reply.get('ok'):
                raise RuntimeError(f"worker inference failed: {reply.get('error')}")
            return reply
        raise TimeoutError(f'worker inference timed out after {self.timeout_s}s')


class PerceptionService(SimObservation):
    """Continuous capture + threaded perception answering requests."""

    def __init__(self, graspnet: PersistentGraspnet, tip_tcp):
        super().__init__()
        self.graspnet = graspnet
        self.tip_tcp = tip_tcp
        self.requests = queue.Queue()
        self.create_subscription(String, '/piperx/perception/request',
                                 self.on_request, 10)
        self.response_pub = self.create_publisher(String, '/piperx/perception/response', 10)
        self.stats = {'requests': 0, 'ok': 0, 'failed': 0,
                      'inference_ms': [], 'total_ms': []}
        self.joints_history = []
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def receive(self, kind, msg):
        """Cache bundles only; heavy work runs in the worker thread."""
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = self.frames.setdefault(stamp, {})
        bundle[kind] = msg
        while len(self.frames) > 32:
            self.frames.popitem(last=False)

    def try_bundle(self, stamp, bundle):  # pragma: no cover - never called
        raise RuntimeError('perception service does not use the base-class capture path')

    def on_joints(self, msg):
        super().on_joints(msg)
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joints_history.append((stamp_s, msg))
        if len(self.joints_history) > 2000:
            del self.joints_history[:1000]

    def on_request(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        self.requests.put(payload)

    def loop(self):
        while True:
            payload = self.requests.get()
            started = time.monotonic()
            response = self.process(payload, started)
            self.response_pub.publish(String(data=json.dumps(response)))

    def process(self, payload, started):
        request_id = payload.get('request_id', -1)
        view_id = str(payload.get('view_id', 'v0'))
        save_dir = Path(payload.get('save_dir') or '/tmp/piperx_perception')
        save_dir.mkdir(parents=True, exist_ok=True)
        timing = {}
        try:
            bundle, stamp, tf, lag = self.wait_bundle(payload.get('max_bundle_age_s', 3.0))
            timing['bundle_wait_ms'] = round((time.monotonic() - started) * 1e3, 1)
            color = decode_color(bundle['color'])
            depth = decode_depth(bundle['depth'])
            info = bundle['info']
            T = transform_from_ros(tf)
            joint_values = dict(zip(self.joints.name, map(float, self.joints.position)))
            stamp_obj = {'sec': int(stamp[0]), 'nanosec': int(stamp[1])}

            frame_npz = save_dir / f'frame_{view_id}.npz'
            if frame_npz.exists():
                frame_npz.unlink()

            model, k, d = camera_model(info)
            yy, xx = np.indices(depth.shape, dtype=np.float32)
            pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
            rays = (cv2_fisheye_or_plain(model, pixels, k, d)
                    ).reshape(depth.shape + (2,))
            z = depth.astype(np.float32) * (self.depth_range / 65535.)
            xyz = np.concatenate((rays * z[..., None], z[..., None]), axis=-1).astype(np.float32)

            selected_pixel = payload.get('target_pixel')
            if selected_pixel is not None:
                # An image-space seed selects a depth-connected surface; no
                # RGB colour is used to choose this target.
                mask = select_depth_component(xyz, selected_pixel)
                target_points = xyz[mask]
            else:
                target_points, mask = color_region_points(color, depth, info, self.depth_range)
            scene_points = depth_points_camera(depth, info, self.depth_range, stride=2)
            t_obs = time.monotonic()
            built = build_generic_observation(
                target_points, scene_points, T, stamp=stamp_obj,
                camera_frame=bundle['depth'].header.frame_id,
                joint_positions_rad=joint_values)
            timing['observation_ms'] = round((time.monotonic() - t_obs) * 1e3, 1)
            if built['status'] != 'ok':
                return self.fail(request_id, view_id, timing,
                                 f"observation rejected: {built['reason']}")
            built['artifacts']['mask'] = mask
            observation = save_generic_observation(save_dir, f'frame_{view_id}', built)
            observation.update(stamp=stamp_obj, tf_lag_s=lag, tf_latest_fallback=False,
                               joint_positions_rad=joint_values,
                               view_id=view_id, T_base_camera=T.tolist(),
                               target_selection=('depth_pixel' if selected_pixel is not None else 'red_color'),
                               target_pixel=selected_pixel)

            np.savez_compressed(frame_npz, xyz=xyz, color=color, depth=depth,
                                mask=mask, k=k, d=d,
                                stamp=np.array([stamp_obj['sec'], stamp_obj['nanosec']],
                                               dtype=np.int64))
            # Inference input follows the offline-demo ROI convention: one
            #独立 ROI per padding (mask bbox + that padding, plus a depth
            # window around the target median). Fusing paddings into one big
            # ROI pulls in table/background points and GraspNet then predicts
            # unusable grasp families.
            roi_paddings = payload.get('roi_paddings') or [40, 60, 80]
            depth_window = float(payload.get('depth_window', 0.08))
            finite = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
            ys, xs = np.nonzero(mask)
            target_median_z = float(np.median(xyz[mask & finite][:, 2])) \
                if (mask & finite).any() else 0.0

            t_inf = time.monotonic()
            candidates = []
            roi_log = []
            # Two sampling seeds per ROI: a single GraspNet pass on one view
            # can predict only upward-scooping grasps (which the wrist
            # precheck correctly rejects); a second sampling pass adds
            # candidate diversity at negligible resident-inference cost.
            for padding in roi_paddings:
                roi = np.ones(xyz.shape[:2], dtype=bool)
                roi[max(0, ys.min() - padding):min(roi.shape[0], ys.max() + padding + 1),
                    max(0, xs.min() - padding):min(roi.shape[1], xs.max() + padding + 1)] = False
                roi = ~roi
                roi &= np.abs(xyz[..., 2] - target_median_z) <= depth_window
                cloud_path = save_dir / f'scene_{view_id}_{padding}.npy'
                np.save(cloud_path, xyz[finite & roi])
                conv_total, rejected_total, worker_ms = [], [], 0.0
                for seed in (42, 43):
                    pred_path = save_dir / f'pred_{view_id}_{padding}_{seed}.npy'
                    reply = self.graspnet.infer(cloud_path, pred_path, seed=seed)
                    worker_ms += reply['time_ms']
                    rows = np.load(reply['predictions'], allow_pickle=False)
                    if rows.ndim != 2 or rows.shape[1] != 17:
                        raise ValueError('worker returned unexpected prediction shape')
                    rows = rows[np.argsort(-rows[:, 0], kind='stable')]
                    target_pts = xyz[mask & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)]
                    conv, rejected = convert_grasps(
                        rows, T, self.tip_tcp[0], self.tip_tcp[1], target_pts)
                    for candidate in conv:
                        candidate['source_roi_padding_px'] = padding
                        candidate['source_seed'] = seed
                    conv_total.extend(conv)
                    rejected_total.extend(rejected)
                candidates.extend(conv_total)
                roi_log.append({'padding_px': padding,
                                'model_predictions': int(len(conv_total) + len(rejected_total)),
                                'accepted': len(conv_total), 'rejected': len(rejected_total),
                                'worker_ms': round(worker_ms, 1)})
            timing['inference_and_convert_ms'] = round((time.monotonic() - t_inf) * 1e3, 1)
            timing['total_ms'] = round((time.monotonic() - started) * 1e3, 1)

            # Annotate + rank through the shared revalidation, cap to 40.
            hull_vertices = None
            if observation.get('convex_hull_path'):
                from piperx_lab.object_observation import read_ply
                hull_vertices, _ = read_ply(save_dir / observation['convex_hull_path'])
            # Revalidation needs the target material in the BASE frame (the
            # candidates and hull already are); color_region_points returns
            # camera-frame points, so transform them here.
            target_points_base = (target_points @ T[:3, :3].T + T[:3, 3])
            candidates = apply_revalidation(candidates[:200], target_points_base,
                                            hull_vertices=hull_vertices,
                                            table_height_m=observation['quality']['table_height_m'])
            # Debug provenance: capture the exact revalidation inputs so a
            # mismatch with offline reproduction is diagnosable.
            np.save(save_dir / f'debug_target_pts_{view_id}.npy', target_points)
            (save_dir / f'debug_convert_{view_id}.json').write_text(json.dumps({
                'T_base_camera': T.tolist(),
                'table_height_m': observation['quality']['table_height_m'],
                'n_target_points': int(len(target_points))}, indent=1))
            valid = [c for c in candidates if c.get('rejection_reason') is None]
            valid.sort(key=lambda c: (-float(c.get('confidence') or 0.0),
                                      float(c.get('approach_distance_m') or 1.0)))
            for candidate_id, candidate in enumerate(valid[:40]):
                candidate['id'] = candidate_id
            rejections = {}
            for candidate in candidates:
                reason = candidate.get('rejection_reason')
                if reason:
                    rejections[reason] = rejections.get(reason, 0) + 1

            self.stats['requests'] += 1
            self.stats['ok'] += 1
            self.stats['inference_ms'].append(timing['inference_and_convert_ms'])
            self.stats['total_ms'].append(timing['total_ms'])
            (save_dir / f'perception_{view_id}.json').write_text(json.dumps({
                'observation': observation, 'candidates': valid[:40],
                'rejections': rejections, 'roi_log': roi_log,
                'timing_ms': timing}, indent=1) + '\n')
            return {'request_id': request_id, 'view_id': view_id, 'ok': True,
                    'observation': observation, 'candidates': valid[:40],
                    'rejections': rejections, 'roi_log': roi_log,
                    'timing_ms': timing, 'frame_npz': str(frame_npz),
                    'stamp': stamp_obj, 'joint_positions_rad': joint_values}
        except Exception as exc:  # noqa: BLE001 - service must keep answering
            return self.fail(request_id, view_id, timing, str(exc))

    def fail(self, request_id, view_id, timing, error):
        self.stats['requests'] += 1
        self.stats['failed'] += 1
        timing['total_ms'] = timing.get('total_ms', round(time.monotonic() * 1e3, 1))
        import traceback
        print(json.dumps({'perception_fail': view_id, 'error': error,
                          'trace': traceback.format_exc()[-800:]}), flush=True)
        return {'request_id': request_id, 'view_id': view_id, 'ok': False,
                'error': error, 'timing_ms': timing}

    def wait_bundle(self, max_age_s):
        """Block for the next synchronised bundle; heavy compute stays here.

        The joint state used is the history entry closest to the image stamp
        (image-time synchronisation, robust to processing backlog).
        """
        deadline = time.monotonic() + max(3.0, max_age_s + 5.0)
        last_error = 'no synchronised bundle'
        while time.monotonic() < deadline:
            bundle_item = self.take_bundle()
            if bundle_item is None:
                time.sleep(0.02)
                continue
            bundle, stamp = bundle_item
            stamp_s = stamp[0] + stamp[1] * 1e-9
            joints_msg = None
            if self.joints_history:
                _, joints_msg = min(self.joints_history,
                                    key=lambda item: abs(item[0] - stamp_s))
            if joints_msg is None:
                last_error = 'no joint feedback'
                time.sleep(0.05)
                continue
            joint_s = joints_msg.header.stamp.sec + joints_msg.header.stamp.nanosec * 1e-9
            try:
                if abs(stamp_s - joint_s) > 0.2:
                    raise TransformException('camera/joint feedback not synchronized')
                tf = self.buffer.lookup_transform(
                    'base_link', 'camera_depth_frame',
                    Time.from_msg(bundle['depth'].header.stamp))
                lag = abs(stamp_s -
                          (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9))
                if lag > 0.2:
                    raise TransformException(f'TF lag {lag:.3f}s')
                return bundle, stamp, tf, lag
            except (TransformException, ValueError) as exc:
                last_error = str(exc)
                time.sleep(0.05)
        raise RuntimeError(f'no usable bundle: {last_error}')

    def take_bundle(self):
        """Pop the oldest complete bundle from the frame cache."""
        for stamp, bundle in list(self.frames.items()):
            if {'color', 'depth', 'info'} <= bundle.keys() and self.depth_range and self.joints:
                del self.frames[stamp]
                return bundle, stamp
        return None


def cv2_fisheye_or_plain(model, pixels, k, d):
    if model == 'fisheye':
        return cv2.fisheye.undistortPoints(pixels, k, d)
    return cv2.undistortPoints(pixels, k, d)


def main():
    parser = __import__('argparse').ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(
        ROOT / 'third_party/graspnet-baseline/checkpoints/checkpoint-rs.tar'))
    args = parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '183':
        raise SystemExit('Requires isolated simulation domain 183')
    rclpy.init()
    tip_model = model_tip_transform(installed_gripper_model_path())
    tip_tcp = (tip_model, np.eye(4))  # T_link6_tip, T_link6_tcp (flange identity)
    graspnet = PersistentGraspnet(checkpoint=args.checkpoint)
    node = PerceptionService(graspnet, tip_tcp)
    node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    # Let the subscription callbacks fill the frame cache before the warm-up
    # inference (the warm-up itself runs synchronously, outside the spin).
    cache_deadline = time.monotonic() + 4.0
    while time.monotonic() < cache_deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    # Warm-up: the first resident inference pays CUDA kernel initialisation
    # (tens of seconds); pay it now so client probes stay fast.
    warmup = node.process({'request_id': 0, 'view_id': 'warmup',
                           'save_dir': '/tmp/piperx_warmup'}, time.monotonic())
    print(json.dumps({'warmup': warmup.get('ok'),
                      'error': warmup.get('error')}), flush=True)
    print('perception service ready', flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stats = dict(node.stats)
        if stats['inference_ms']:
            stats['inference_ms_mean'] = round(sum(stats['inference_ms']) / len(stats['inference_ms']), 1)
            stats['total_ms_mean'] = round(sum(stats['total_ms']) / len(stats['total_ms']), 1)
        print(json.dumps({'perception_stats': stats}), flush=True)
        node.destroy_node()
        rclpy.shutdown()
        try:
            graspnet.proc.terminate()
        except OSError:
            pass


if __name__ == '__main__':
    main()
