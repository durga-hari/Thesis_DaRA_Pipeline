from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


class YOLOv8Detector:
    """Ultralytics YOLO (v8 weights or later), person class only."""

    def __init__(self, weights: str = "yolov8m.pt", device: str = "cuda", conf: float = 0.25,
                 iou: float = 0.5, imgsz: int = 640, max_det: int = 300,
                 half: Optional[bool] = None, classes: Sequence[int] = (0,)):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError("Ultralytics is required: pip install ultralytics") from e

        import torch
        self.device = device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.conf, self.iou, self.imgsz, self.max_det = float(conf), float(iou), int(imgsz), int(max_det)
        self.half = (self.device != "cpu") if half is None else bool(half and self.device != "cpu")
        self.classes = list(classes)
        self.model = YOLO(str(weights))
        log.info("YOLOv8Detector: %s (device=%s conf=%.2f iou=%.2f imgsz=%d)",
                 weights, self.device, self.conf, self.iou, self.imgsz)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        res = self.model.predict(frame_bgr, verbose=False, device=self.device, classes=self.classes,
                                 conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 max_det=self.max_det, half=self.half)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        b = res[0].boxes
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)
        conf = b.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
        return np.hstack([xyxy, conf])
