"""
Lightweight StrongSORT-style tracker.

- constant-velocity Kalman filter on [cx, cy, w, h]
- IoU gate, then cosine appearance similarity, greedy per detection
- unmatched detections start new tracks immediately (no tentative state)
- tracks are dropped after `ttl` frames without a match

Note for write-ups: this is greedy association, not the Hungarian matching /
EMA / camera-motion compensation of the full StrongSORT paper.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ub = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (ua + ub - inter + 1e-12))


class _KF:
    """Minimal 8D state Kalman filter: [cx, cy, w, h, vx, vy, vw, vh]."""

    def __init__(self, box):
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        self.x = np.array([cx, cy, box[2] - box[0], box[3] - box[1], 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 1.0
        self.H = np.eye(4, 8, dtype=np.float32)
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = 1.0

    def predict(self):
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q

    def update(self, box):
        z = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0,
                      box[2] - box[0], box[3] - box[1]], dtype=np.float32)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P

    def box(self):
        cx, cy, w, h = self.x[0], self.x[1], max(1.0, self.x[2]), max(1.0, self.x[3])
        return [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]


class StrongSORTTracker:
    def __init__(self, sim_thr: float = None, iou_thr: float = None, ttl: int = None,
                 max_cosine_distance: float = 0.45, max_iou_distance: float = 0.3,
                 max_age: int = 60, **_):
        # The DeepSORT-style key names are accepted as aliases so one config /
        # parameter sweep can drive either tracker (same mapping as before).
        self.sim_thr = float(sim_thr if sim_thr is not None else max_cosine_distance)
        self.iou_thr = float(iou_thr if iou_thr is not None else max_iou_distance)
        self.ttl = int(ttl if ttl is not None else max_age)
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}

    def update(self, boxes, scores, feats) -> List[dict]:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for t in self._tracks.values():
            t["kf"].predict()
            t["age"] += 1
        for tid in [tid for tid, t in self._tracks.items() if t["age"] > self.ttl]:
            self._tracks.pop(tid)
        n = boxes.shape[0]
        if n == 0:
            return []
        feats = [None] * n if feats is None else [np.asarray(f, dtype=np.float32) for f in feats]

        det_to_tid = [-1] * n
        used = set()
        for i, box in enumerate(boxes):
            best_id, best = None, -1e9
            for tid, t in self._tracks.items():
                if tid in used or _iou(box, t["kf"].box()) < self.iou_thr:
                    continue
                s = -1.0 if (feats[i] is None or t["feat"] is None) else float(np.dot(feats[i], t["feat"]))
                if s > best:
                    best, best_id = s, tid
            if best_id is not None and best >= self.sim_thr:
                t = self._tracks[best_id]
                t["kf"].update(box)
                if feats[i] is not None:
                    t["feat"] = feats[i]
                t["age"] = 0
                det_to_tid[i] = best_id
                used.add(best_id)

        for i, box in enumerate(boxes):
            if det_to_tid[i] == -1:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"kf": _KF(box), "feat": feats[i], "age": 0}
                det_to_tid[i] = tid

        return [{"id": int(tid), "bbox": [float(v) for v in boxes[i]], "det_index": i}
                for i, tid in enumerate(det_to_tid)]
