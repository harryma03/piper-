#!/usr/bin/env python3
"""Read-only RGB-D point-cloud viewer for Gazebo, C1W, and saved frames.

Run with ROS sourced: python3 scripts/pointcloud_viewer.py --source gazebo
Use --source real for the C1W SDK cloud; no robot command is ever published.
"""
import argparse
from collections import OrderedDict
import math
import os
from pathlib import Path
import sys
import threading
import subprocess
import tempfile
import json
import uuid
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/piperx_lab'))
sys.path.insert(0, str(ROOT / 'scripts'))

from piperx_lab.camera_target import decode_color, decode_depth, largest_color_mask
from piperx_lab.handeye import camera_model
from piperx_lab.depth_selection import select_depth_component
from diagnose_cylinder_geometry import organized_sdk_cloud

app = Flask(__name__)
shared = {'frame': None, 'error': '等待相机图像', 'updated': 0.0, 'sequence': 0}
frame_cache = OrderedDict()
MAX_CACHED_FRAMES = 12
lock = threading.Lock()
detection_lock = threading.Lock()
detection_cache = OrderedDict()


def camera_xyz(depth, info, depth_range):
    """Reproject aligned optical-Z depth with the advertised camera model."""
    if depth.shape != (info.height, info.width) or not math.isfinite(depth_range) or depth_range <= 0:
        raise ValueError('深度图、内参或量程无效')
    model, k, d = camera_model(info)
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    rays = (cv2.fisheye.undistortPoints(pixels, k, d) if model == 'fisheye'
            else cv2.undistortPoints(pixels, k, d)).reshape(depth.shape + (2,))
    z = depth.astype(np.float32) * (depth_range / 65535.0)
    return np.concatenate((rays * z[..., None], z[..., None]), axis=-1)


def create_live_node(source):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from std_msgs.msg import Float32

    class CameraReader(Node):
        def __init__(self):
            super().__init__('piper_pointcloud_viewer')
            if source == 'gazebo':
                self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
            self.bundles = OrderedDict()
            self.depth_range = None
            self.last_process = 0.0
            topics = [('color', Image, '/camera/color/aligned'),
                      ('depth', Image, '/camera/depth'),
                      ('info', CameraInfo, '/camera/info/depth')]
            if source == 'real':
                topics.append(('cloud', PointCloud2, '/camera/pointcloud'))
            for kind, typ, topic in topics:
                self.create_subscription(typ, topic, lambda m, k=kind: self.receive(k, m), 10)
            self.create_subscription(Float32, '/camera/depth_range', self.on_range, 10)

        def on_range(self, msg):
            if math.isfinite(msg.data) and msg.data > 0:
                self.depth_range = float(msg.data)

        def receive(self, kind, msg):
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            bundle = self.bundles.setdefault(key, {})
            bundle[kind] = msg
            while len(self.bundles) > 12:
                self.bundles.popitem(last=False)
            required = {'color', 'depth', 'info'} | ({'cloud'} if source == 'real' else set())
            if not required <= bundle.keys() or self.depth_range is None:
                return
            del self.bundles[key]
            now = time.monotonic()
            if now - self.last_process < 0.8:
                return
            self.last_process = now
            try:
                color, depth, info = decode_color(bundle['color']), decode_depth(bundle['depth']), bundle['info']
                if color.shape[:2] != depth.shape or depth.shape != (info.height, info.width):
                    raise ValueError('彩色图、深度图和内参尺寸不一致')
                xyz = (organized_sdk_cloud(bundle['cloud'], depth) if source == 'real'
                       else camera_xyz(depth, info, self.depth_range))
                frame = {'color': color.copy(), 'xyz': xyz, 'mask': None,
                         'label': '实时 C1W SDK 点云' if source == 'real' else '实时 Gazebo RGB-D',
                         'stamp': [int(key[0]), int(key[1])]}
                with lock:
                    shared['sequence'] += 1
                    frame['frame_id'] = str(shared['sequence'])
                    shared.update(frame=frame, error=None, updated=now)
                    frame_cache[frame['frame_id']] = (frame, now)
                    while len(frame_cache) > MAX_CACHED_FRAMES:
                        frame_cache.popitem(last=False)
            except Exception as exc:
                with lock:
                    shared['error'] = str(exc)

    rclpy.init()
    node = CameraReader()
    def spin():
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass
        except Exception:
            if rclpy.ok():
                raise
        finally:
            node.destroy_node()
    node.viewer_thread = threading.Thread(target=spin, daemon=True)
    node.viewer_thread.start()
    return node


def saved_frames():
    root = ROOT / 'records'
    paths = []
    for p in root.rglob('frame*.npz'):
        if 'object_points' in p.name or not p.is_file():
            continue
        try:
            paths.append((p.stat().st_mtime, p))
        except OSError:
            continue
    paths.sort(reverse=True)
    real = root / 'cylinder_real/geometry_diagnostic/frame.npz'
    selected = [p for _, p in paths[:80]]
    if real.is_file():
        if real in selected:
            selected.remove(real)
        selected.insert(0, real)
    return [{'id': str(p.relative_to(root)), 'label': str(p.relative_to(root))} for p in selected]


def load_frame(source, frame_id=None):
    if source == 'live':
        with lock:
            error = shared['error']
            if frame_id:
                entry = frame_cache.get(frame_id)
                if entry is None and shared.get('pinned_id') == frame_id:
                    entry = shared.get('pinned_frame')
                if entry is None:
                    raise ValueError('图像帧已过期，请刷新后重新点选')
                frame, updated = entry
            else:
                frame, updated = shared['frame'], shared['updated']
        if frame is None:
            raise ValueError(error or '等待同步相机图像')
        return frame, time.monotonic() - updated, error
    root = (ROOT / 'records').resolve()
    path = (root / source).resolve()
    if root not in path.parents or not path.is_file() or path.suffix != '.npz':
        raise ValueError('记录文件不存在')
    with np.load(path, allow_pickle=False) as data:
        if not {'xyz', 'color'} <= set(data.files):
            raise ValueError('记录缺少彩色图或点云')
        frame = {'xyz': np.asarray(data['xyz']), 'color': np.asarray(data['color']),
                 'mask': np.asarray(data['mask'], bool) if 'mask' in data else None,
                 'stamp': np.asarray(data['stamp']).tolist() if 'stamp' in data else None,
                 'label': '历史记录：' + source}
    return frame, None, None


def render_data(frame, color_choice, pixel=None, detection_mask=None):
    xyz = np.asarray(frame['xyz'])
    color = np.asarray(frame['color'])
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or color.shape != xyz.shape:
        raise ValueError('点云和彩色图尺寸不匹配')
    if color_choice == 'yolo':
        mask = detection_mask if detection_mask is not None else np.zeros(xyz.shape[:2], bool)
        if mask.shape != xyz.shape[:2]:
            raise ValueError('识别掩码与点云尺寸不一致')
    elif color_choice == 'point':
        mask = (np.zeros(xyz.shape[:2], dtype=bool) if pixel is None
                else select_depth_component(xyz, pixel))
    elif color_choice == 'saved':
        mask = frame.get('mask')
        if mask is None:
            raise ValueError('实时图像没有历史掩码，请选择一种颜色')
    else:
        mask = largest_color_mask(color, color_choice, 80)
        if mask is None:
            mask = np.zeros(xyz.shape[:2], bool)
        else:
            mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    valid = np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0)
    mask = np.asarray(mask, bool) & valid
    target = xyz[mask]
    center = np.median(target, axis=0).tolist() if len(target) else None
    # Uniform subsampling preserves the whole field without sending a huge cloud.
    flat = xyz.reshape(-1, 3)
    colors = color.reshape(-1, 3)[:, ::-1]
    good = valid.ravel()
    ids = np.flatnonzero(good)
    if len(ids) > 9000:
        ids = ids[np.linspace(0, len(ids) - 1, 9000, dtype=int)]
    target_ids = np.flatnonzero(mask.ravel())
    if len(target_ids) > 4000:
        target_ids = target_ids[np.linspace(0, len(target_ids) - 1, 4000, dtype=int)]
    def values(indices):
        return np.round(flat[indices], 4).tolist()
    return {'scene': values(ids), 'scene_rgb': colors[ids].tolist(),
            'target': values(target_ids), 'center': center,
            'target_count': int(len(target)), 'valid_count': int(valid.sum()),
            'label': frame['label'], 'stamp': frame.get('stamp')}, mask


def selected_pixel():
    if request.args.get('color') != 'point':
        return None
    if 'x' not in request.args and 'y' not in request.args:
        return None
    try:
        return [int(request.args['x']), int(request.args['y'])]
    except (KeyError, ValueError):
        raise ValueError('请先点击左侧图像中的目标')


@app.get('/')
def index():
    return render_template_string((ROOT / 'scripts/pointcloud_viewer.html').read_text())


@app.get('/api/frames')
def frames():
    return jsonify(saved_frames())


def requested_detection(source, frame):
    if request.args.get('color') != 'yolo':
        return None, None
    token = request.args.get('detection_id')
    if not token:
        return None, None
    with detection_lock:
        entry = detection_cache.get(token)
    if not entry or entry['source'] != source or entry['frame_id'] != frame.get('frame_id'):
        raise ValueError('识别结果已过期或与图像帧不匹配，请重新识别')
    try:
        idx = int(request.args.get('instance', '0'))
        if idx < 0 or idx >= len(entry['items']):
            raise ValueError()
        return entry['masks'][idx], entry['items'][idx]
    except (ValueError, IndexError):
        raise ValueError('请选择有效的识别目标')


@app.post('/api/detect')
def detect():
    try:
        req = request.get_json() or {}
        source = req.get('source', 'live')
        frame_id = req.get('frame_id')
        if source == 'live' and not frame_id:
            raise ValueError('请等待图像加载后再识别')
        frame, age, _ = load_frame(source, frame_id)
        if source == 'live':
            with lock:
                shared['pinned_id'] = frame_id
                shared['pinned_frame'] = (frame, time.monotonic() - age)
        python = ROOT / '.graspnet-venv/bin/python'
        model = ROOT / 'third_party/yolo_models/yolo11n-seg.pt'
        if not python.exists() or not model.exists():
            raise ValueError('YOLO 运行环境或模型未安装，请查看文档')
        with tempfile.TemporaryDirectory(prefix='piper-yolo-') as tmp:
            image_path, output = Path(tmp)/'frame.png', Path(tmp)/'masks.npz'
            if not cv2.imwrite(str(image_path), frame['color']):
                raise ValueError('保存识别图像失败')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(ROOT / 'third_party/yolo_runtime')
            env['YOLO_CONFIG_DIR'] = str(ROOT / 'tmp/yolo_config')
            run = subprocess.run([str(python), str(ROOT / 'scripts/yolo_seg_worker.py'),
                                  str(image_path), str(output), str(model)], env=env,
                                 capture_output=True, text=True, timeout=120)
            if run.returncode:
                raise ValueError('模型运行失败：' + run.stderr[-500:])
            items = json.loads(run.stdout.strip().splitlines()[-1])
            with np.load(output, allow_pickle=False) as data:
                masks = np.asarray(data['masks'], bool)
        token = uuid.uuid4().hex
        with detection_lock:
            detection_cache[token] = {'source': source, 'frame_id': frame.get('frame_id'),
                                      'items': items, 'masks': masks}
            while len(detection_cache) > 4:
                detection_cache.popitem(last=False)
        return jsonify({'detection_id': token, 'frame_id': frame.get('frame_id'), 'items': items})
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as exc:
        return jsonify({'error': str(exc)}), 400


@app.get('/api/cloud')
def cloud():
    try:
        frame, age, error = load_frame(request.args.get('source', 'live'), request.args.get('frame_id'))
        pixel = selected_pixel()
        detection_mask, _ = requested_detection(request.args.get('source', 'live'), frame)
        data, _ = render_data(frame, request.args.get('color', 'red'), pixel, detection_mask)
        if pixel is not None and request.args.get('source', 'live') == 'live' and frame.get('frame_id'):
            with lock:
                shared['pinned_id'] = frame['frame_id']
                shared['pinned_frame'] = (frame, time.monotonic() - age)
        data['age_s'] = round(age, 1) if age is not None else None
        data['frame_id'] = frame.get('frame_id')
        data['warning'] = error
        return jsonify(data)
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({'error': str(exc)}), 400


@app.get('/api/image')
def image():
    try:
        frame, _, _ = load_frame(request.args.get('source', 'live'), request.args.get('frame_id'))
        detection_mask, item = requested_detection(request.args.get('source', 'live'), frame)
        _, mask = render_data(frame, request.args.get('color', 'red'), selected_pixel(), detection_mask)
        image = frame['color'].copy()
        overlay = image.copy()
        overlay[mask] = (0, 255, 0)
        image = cv2.addWeighted(image, .65, overlay, .35, 0)
        if item:
            x1, y1, x2, y2 = item['box']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, f"{item['name']} {item['confidence']:.2f}", (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError('图片编码失败')
        return send_file(BytesIO(buf.tobytes()), mimetype='image/jpeg', max_age=0)
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({'error': str(exc)}), 400


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('gazebo', 'real', 'file'), default='gazebo')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.source == 'gazebo' and os.environ.get('ROS_DOMAIN_ID') != '183':
        raise SystemExit('Gazebo viewer requires ROS_DOMAIN_ID=183')
    node = create_live_node(args.source) if args.source != 'file' else None
    print(f'Point-cloud viewer: http://127.0.0.1:{args.port} (source: {args.source})', flush=True)
    try:
        app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)
    finally:
        if node is not None:
            import rclpy
            rclpy.try_shutdown()
            node.viewer_thread.join(timeout=3)


if __name__ == '__main__':
    main()
