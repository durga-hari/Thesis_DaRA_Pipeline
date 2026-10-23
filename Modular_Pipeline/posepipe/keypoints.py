"""
Keypoint formats shared by all pose backends.

  coco17          17 body joints (AlphaPose HRNet, RTMPose body models)
  coco_wholebody  133 joints; indices 0..16 are exactly the COCO-17 body joints,
                  followed by 6 feet, 68 face, 21 left-hand, 21 right-hand.

Because the first 17 WholeBody joints *are* COCO-17, converting WholeBody ->
COCO-17 is a slice. This is what makes AlphaPose and RTMPose comparable on the
same joints (`pipeline.keypoint_format: coco17`).
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = {"coco17": 17, "coco_wholebody": 133}

COCO17_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Body + head + hand chains (as used for the RTMO/RTMPose visualisations)
WHOLEBODY_EDGES = COCO17_EDGES + [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (91, 92), (92, 93), (93, 94), (94, 95),
    (96, 97), (97, 98), (98, 99), (99, 100),
    (101, 102), (102, 103), (103, 104), (104, 105),
    (106, 107), (107, 108), (108, 109), (109, 110),
    (111, 112), (112, 113), (113, 114), (114, 115),
]


def format_for(num_joints: int) -> str:
    if num_joints == 17:
        return "coco17"
    if num_joints == 133:
        return "coco_wholebody"
    return f"custom{num_joints}"


def edges_for(num_joints: int):
    return WHOLEBODY_EDGES if num_joints >= 133 else COCO17_EDGES


def to_xyc(kps) -> np.ndarray:
    """Coerce (J,2) / (J,3) / (1,J,3) / flat into (J,3) float32 [x, y, conf]."""
    a = np.asarray(kps, dtype=np.float32)
    if a.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    if a.ndim == 1:
        a = a.reshape(-1, 3)
    if a.shape[1] == 2:
        a = np.hstack([a, np.ones((a.shape[0], 1), dtype=np.float32)])
    return a[:, :3]


def convert(kps: np.ndarray, target: str) -> np.ndarray:
    """Convert (J,3) keypoints to `target` ('native' returns them unchanged)."""
    if target in (None, "", "native"):
        return kps
    if target not in NUM_JOINTS:
        raise ValueError(f"Unknown keypoint_format '{target}'. Use native|coco17|coco_wholebody")
    j_out = NUM_JOINTS[target]
    j_in = kps.shape[0]
    if j_in == j_out:
        return kps
    if j_in > j_out:
        return kps[:j_out]
    # Upcast (e.g. COCO-17 -> WholeBody): missing joints get conf 0.
    pad = np.zeros((j_out - j_in, 3), dtype=np.float32)
    return np.vstack([kps, pad])
