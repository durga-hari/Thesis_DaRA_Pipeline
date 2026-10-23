"""
RTMPose (MMPose 1.x) as a top-down backend, e.g. RTMPose-L COCO-WholeBody 384x288.

Uses mmpose.apis.init_model + inference_topdown directly (no MMDet needed).
MMPose's data preprocessor expects BGR input (it does bgr_to_rgb itself), so
frames are passed as read by OpenCV. Set `input_color: rgb` only to reproduce
the old RTMO_DeepSORT_Pipeline results, which fed RGB frames.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

log = logging.getLogger(__name__)


class RTMPoseEstimator:
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda", input_color: str = "bgr", **_):
        try:
            from mmpose.apis import inference_topdown, init_model
        except Exception as e:
            raise ImportError("MMPose 1.x is required for pose.type=rtmpose "
                              "(see README: mim install mmengine mmcv mmpose)") from e
        import torch

        if not (str(device).startswith("cuda") and torch.cuda.is_available()):
            device = "cpu"
        self._infer = inference_topdown
        self.model = init_model(str(cfg), str(ckpt), device=device)
        self.input_color = str(input_color).lower()
        meta = getattr(self.model, "dataset_meta", None) or {}
        self.num_joints = int(meta.get("num_keypoints", 133))
        log.info("RTMPose ready: %s (%d joints, device=%s, input=%s)",
                 cfg, self.num_joints, device, self.input_color)

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy) -> List[dict]:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4) if len(boxes_xyxy) else np.empty((0, 4))
        if boxes.shape[0] == 0:
            return []
        img = frame_bgr[:, :, ::-1].copy() if self.input_color == "rgb" else frame_bgr
        samples = self._infer(self.model, img, bboxes=boxes, bbox_format="xyxy")

        out: List[dict] = []
        for sample in samples:  # one PoseDataSample per input box, same order
            inst = sample.pred_instances
            xy = np.asarray(inst.keypoints, dtype=np.float32).reshape(-1, self.num_joints, 2)[0]
            sc = np.asarray(inst.keypoint_scores, dtype=np.float32).reshape(-1, self.num_joints)[0]
            sc = np.clip(sc, 0.0, 1.0)
            out.append({"keypoints": np.concatenate([xy, sc[:, None]], axis=1),
                        "score": float(sc.mean())})
        return out
