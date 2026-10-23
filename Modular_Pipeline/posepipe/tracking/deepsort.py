"""
DeepSORT (Wojke et al.) using the vendored core in posepipe/third_party/deep_sort.

The detection index is carried through DeepSORT via Detection.mask (the core
copies it onto the matched Track), so every output track knows which detection
it was updated with this frame.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..third_party.deep_sort.detection import Detection
from ..third_party.deep_sort.nn_matching import NearestNeighborDistanceMetric
from ..third_party.deep_sort.tracker import Tracker


class DeepSortTracker:
    def __init__(self, max_cosine_distance: float = 0.2, max_iou_distance: float = 0.7,
                 max_age: int = 30, n_init: int = 3, nn_budget: Optional[int] = None, **_):
        metric = NearestNeighborDistanceMetric("cosine", float(max_cosine_distance), nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=float(max_iou_distance),
                               max_age=int(max_age), n_init=int(n_init))

    def update(self, boxes, scores, feats) -> List[dict]:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        detections = []
        if boxes.shape[0]:
            if feats is None or len(feats) != boxes.shape[0]:
                raise ValueError("DeepSORT needs one ReID feature per detection.")
            feats = np.asarray(feats, dtype=np.float32)
            for i, (b, s, f) in enumerate(zip(boxes, scores, feats)):
                tlwh = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                detections.append(Detection(tlwh, float(s), 1, f, mask=i))

        self.tracker.predict()
        self.tracker.update(detections)  # also ages / deletes tracks on empty frames

        out = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in t.to_tlbr())
            out.append({"id": int(t.track_id), "bbox": [x1, y1, x2, y2], "det_index": int(t.mask)})
        return out
