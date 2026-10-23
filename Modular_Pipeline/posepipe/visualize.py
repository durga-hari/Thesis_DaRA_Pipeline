from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .keypoints import edges_for, to_xyc


def color_for_id(id_num: int, mode: str = "local"):
    salt = 1 if mode == "local" else 99991
    hue = ((int(id_num) * 37 * salt) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_pose_box_and_id(frame, keypoints, tid, color=None, bbox=None, label=None,
                         conf_thr: float = 0.3):
    """Draw box, skeleton and ID. The skeleton is chosen from the joint count
    (COCO-17 or COCO-WholeBody), so it works for every pose backend."""
    kps = to_xyc(keypoints) if keypoints is not None else np.zeros((0, 3), np.float32)
    if color is None:
        color = color_for_id(tid)
    text = label or f"ID {tid}"

    if bbox is not None:
        b = np.asarray(bbox).flatten().astype(int)
        if b.size >= 4:
            x1, y1, x2, y2 = b[:4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if kps.shape[0] == 0:
        return frame

    J = kps.shape[0]
    for i, j in edges_for(J):
        if i < J and j < J and kps[i, 2] > conf_thr and kps[j, 2] > conf_thr:
            cv2.line(frame, (int(kps[i, 0]), int(kps[i, 1])), (int(kps[j, 0]), int(kps[j, 1])),
                     color, 1, lineType=cv2.LINE_AA)

    for idx, (x, y, c) in enumerate(kps):
        if c > conf_thr:
            small = J >= 133 and 23 <= idx <= 90  # WholeBody face points
            cv2.circle(frame, (int(x), int(y)), 1 if small else 2,
                       (0, 200, 200) if small else color, -1)

    if bbox is None:
        valid = kps[kps[:, 2] > conf_thr]
        if valid.size:
            cv2.putText(frame, text, (int(valid[:, 0].min()), int(valid[:, 1].min()) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame
