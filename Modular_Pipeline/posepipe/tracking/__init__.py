"""
Per-camera multi-object trackers.

Interface (every tracker):

    tracks = tracker.update(boxes, scores, feats)
      boxes:  (N, 4) [x1, y1, x2, y2]
      scores: (N,) detection confidences
      feats:  (N, D) L2-normalised ReID embeddings
    -> list of {"id": int, "bbox": [x1, y1, x2, y2], "det_index": int}

`det_index` is the row of the detection the track was matched to in this
frame, so keypoints / features / scores are taken from the right person.
A new tracker is built for every video.
"""

from ..registry import Registry

TRACKERS = Registry("tracker")
TRACKERS.register_lazy("deepsort", "posepipe.tracking.deepsort:DeepSortTracker", aliases=("deep_sort",))
TRACKERS.register_lazy("strongsort", "posepipe.tracking.strongsort:StrongSORTTracker",
                       aliases=("strong_sort",))


def build_tracker(cfg: dict):
    return TRACKERS.build(cfg)
