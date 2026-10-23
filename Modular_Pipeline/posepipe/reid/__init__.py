"""
ReID appearance extractors.

Interface:  reid(frame_bgr, boxes_xyxy) -> np.ndarray (N, D), L2-normalised rows
"""

from ..registry import Registry

REID = Registry("reid")
REID.register_lazy("osnet", "posepipe.reid.torchreid_extractor:TorchreidExtractor",
                   aliases=("osnet_x1_0",),
                   defaults={"model_name": "osnet_x1_0", "num_classes": 1041, "imagenet_init": False})
REID.register_lazy("resnet50", "posepipe.reid.torchreid_extractor:TorchreidExtractor",
                   aliases=("resnet",),
                   defaults={"model_name": "resnet50", "num_classes": 751, "imagenet_init": True})


def build_reid(cfg: dict):
    t = str(cfg.get("type", "")).lower()
    # Any other torchreid OSNet variant (osnet_x0_25, osnet_ain_x1_0, ...)
    if t.startswith("osnet") and t not in ("osnet", "osnet_x1_0"):
        return REID.build({**cfg, "type": "osnet", "model_name": cfg.get("model_name", t)})
    return REID.build(cfg)
