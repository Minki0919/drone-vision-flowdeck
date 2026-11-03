"""V2 inference with visible-curve support to reject wall/vehicle boxes."""
import os
import time
from pathlib import Path

import cv2
import numpy as np


class MissionYOLO:
    def __init__(self, root, device='cpu'):
        root = Path(root)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(root/'.cache/ultralytics'))
        from ultralytics import YOLO
        self.model = YOLO(str(root/'models/mission_yolo_v3/rings.pt'))
        if 'ring' not in self.model.names.values():
            raise ValueError('Mission YOLO requires a ring class')
        self.device = device
        self.last_time = None
        self.last = []
        self.rejected_rings = 0
        self.inference_ms = None
        self.predict(np.zeros((320, 320), np.uint8), 0.)
        self.last_time, self.last = None, []

    def predict(self, gray, stamp):
        if gray.ndim != 2 or gray.dtype != np.uint8 or not np.isfinite(stamp):
            raise ValueError('Mission YOLO requires uint8 grayscale and a finite capture time')
        if self.last_time is not None and 0 <= stamp-self.last_time < .10-1e-9:
            return self.last
        start = time.perf_counter()
        result = self.model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            imgsz=320, conf=.40, device=self.device, verbose=False)[0]
        self.last = []
        self.rejected_rings = 0
        for row in result.boxes.data.cpu().numpy():
            label = self.model.names[int(row[5])]
            if label == 'ring' and not self.ring_supported(gray, row[:4]):
                self.rejected_rings += 1
                continue
            self.last.append(dict(model='rings' if label == 'ring' else 'arrows', label=label, confidence=float(row[4]),
                xyxy=row[:4].tolist(), capture_seconds=float(stamp)))
        self.last_time = stamp
        self.inference_ms = (time.perf_counter()-start)*1000
        return self.last

    @staticmethod
    def ring_supported(gray, box):
        """A YOLO ring needs an observed curved rim covering its box."""
        h, w = gray.shape
        box = np.asarray(box, float)
        if not np.isfinite(box).all() or min(box[2:]-box[:2]) < 6:
            return False
        x1, y1 = np.maximum(0, np.floor(box[:2]-4)).astype(int)
        x2, y2 = np.minimum([w, h], np.ceil(box[2:]+4)).astype(int)
        if x2 <= x1 or y2 <= y1:
            return False
        contours = cv2.findContours(cv2.Canny(gray[y1:y2, x1:x2], 40, 100),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        arcs = []
        for contour in contours:
            arcs.append(contour)
            # Occluders can join a circular edge to a straight wall edge.
            # Also fit overlapping measured arcs instead of requiring a full ring.
            if len(contour) >= 48:
                size = len(contour)//2
                arcs += [contour[i:i+size] for i in (0, len(contour)//4, len(contour)//2)]
        for arc in arcs:
            if len(arc) < 16:
                continue
            (cx, cy), (a, b), degrees = cv2.fitEllipse(arc)
            if (not np.isfinite([cx, cy, a, b, degrees]).all()
                    or min(a, b) < 6 or min(a, b)/max(a, b) < .2):
                continue
            theta = np.deg2rad(degrees)
            local = (arc.reshape(-1, 2)-[cx, cy]) @ np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            normalized = local/[a/2, b/2]
            # Rectangles can be fitEllipse inputs too; reject their radial error.
            if np.quantile(np.abs(np.linalg.norm(normalized, axis=1)-1.), .7) > .09:
                continue
            angles = np.arctan2(normalized[:, 1], normalized[:, 0])
            bins = np.unique(np.floor((angles+np.pi)/(2*np.pi)*36).astype(int))
            if len(bins) < 14:
                continue
            extent = np.sqrt([(a/2*np.cos(theta))**2+(b/2*np.sin(theta))**2,
                              (a/2*np.sin(theta))**2+(b/2*np.cos(theta))**2])
            lo = np.maximum(0., np.array([cx+x1, cy+y1])-extent)
            hi = np.minimum([w, h], np.array([cx+x1, cy+y1])+extent)
            area = np.prod(np.maximum(0., hi-lo))
            intersection = np.prod(np.maximum(0., np.minimum(hi, box[2:])-np.maximum(lo, box[:2])))
            union = area+np.prod(box[2:]-box[:2])-intersection
            # A small round window inside a large wall box is insufficient.
            if union > 0 and intersection/union >= .45:
                return True
        return False
