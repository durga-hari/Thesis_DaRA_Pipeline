"""
Person detectors.

Interface:  detector(frame_bgr) -> np.ndarray (N, 5) [x1, y1, x2, y2, score]

The detector is a separate stage (not hidden inside a pose model) so every
pose backend receives exactly the same boxes in a comparison.
"""

from ..registry import Registry

DETECTORS = Registry("detector")
DETECTORS.register_lazy("yolov8", "posepipe.detection.yolov8:YOLOv8Detector",
                        aliases=("yolo", "ultralytics"))


def build_detector(cfg: dict):
    return DETECTORS.build(cfg)
