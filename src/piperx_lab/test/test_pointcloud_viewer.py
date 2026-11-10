"""Verify a clicked live image and its point cloud use the same snapshot."""
import importlib.util
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('pointcloud_viewer_test', ROOT / 'scripts/pointcloud_viewer.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_clicked_frame_remains_available_and_keeps_image_cloud_aligned():
    y, x = np.mgrid[:100, :120]
    xyz = np.stack((x * .002, y * .002, np.full_like(x, .8, dtype=float)), axis=-1)
    xyz[25:65, 30:70, 2] = .5
    old = {'xyz': xyz, 'color': np.zeros((100, 120, 3), np.uint8),
           'mask': None, 'label': 'test old', 'stamp': [1, 0], 'frame_id': 'old'}
    new = dict(old, label='test new', stamp=[2, 0], frame_id='new')
    now = time.monotonic()
    with viewer.lock:
        viewer.frame_cache.clear()
        viewer.frame_cache.update({'old': (old, now), 'new': (new, now)})
        viewer.shared.update(frame=new, updated=now, error=None)
    client = viewer.app.test_client()
    response = client.get('/api/cloud?source=live&color=point&x=45&y=40&frame_id=old')
    assert response.status_code == 200
    assert response.json['frame_id'] == 'old'
    assert response.json['target_count'] == 1600
    with viewer.lock:
        viewer.frame_cache.pop('old')  # pinned selection survives normal eviction
    image = client.get('/api/image?source=live&color=point&x=45&y=40&frame_id=old')
    assert image.status_code == 200
    assert image.mimetype == 'image/jpeg'
    expired = client.get('/api/cloud?source=live&color=point&frame_id=unknown')
    assert expired.status_code == 400


def test_yolo_mask_selects_only_same_frame_points():
    xyz = np.ones((10, 12, 3), dtype=float)
    xyz[:, :, 2] = .8
    xyz[2:5, 3:7, 2] = .4
    mask = np.zeros((10, 12), bool)
    mask[2:5, 3:7] = True
    frame = {'xyz': xyz, 'color': np.zeros((10, 12, 3), np.uint8),
             'label': 'test', 'frame_id': 'a'}
    now = time.monotonic()
    with viewer.lock:
        viewer.frame_cache.update({'a': (frame, now), 'b': (dict(frame, frame_id='b'), now)})
    with viewer.detection_lock:
        viewer.detection_cache['test-token'] = {'source': 'live', 'frame_id': 'a',
            'masks': np.array([mask]), 'items': [{'name': 'test', 'confidence': .9, 'box': [3, 2, 7, 5]}]}
    client = viewer.app.test_client()
    query = 'source=live&color=yolo&detection_id=test-token&instance=0'
    selected = client.get('/api/cloud?' + query + '&frame_id=a')
    assert selected.status_code == 200
    assert selected.json['target_count'] == 12
    assert abs(selected.json['center'][2] - .4) < 1e-6
    assert client.get('/api/image?' + query + '&frame_id=a').status_code == 200
    assert client.get('/api/cloud?' + query + '&frame_id=b').status_code == 400
