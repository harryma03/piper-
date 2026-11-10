#!/usr/bin/env python3
"""One-shot local YOLO segmentation worker. No robot or ROS access."""
import contextlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/piper-yolo-config')
import cv2
import numpy as np

image_path, output_path, model_path = sys.argv[1:4]
with contextlib.redirect_stdout(sys.stderr):
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.25, retina_masks=True, verbose=False)
r = results[0]
image = cv2.imread(image_path)
items, masks = [], []
if r.masks is not None:
    for box, mask in zip(r.boxes, r.masks.data):
        m = mask.cpu().numpy()
        if m.shape != image.shape[:2]:
            m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(m > 0.5)
        cls = int(box.cls.item())
        items.append({'name': str(r.names[cls]), 'confidence': float(box.conf.item()),
                      'box': [int(x) for x in box.xyxy[0].cpu().tolist()]})
np.savez_compressed(output_path, masks=np.asarray(masks, dtype=bool))
print(json.dumps(items))
