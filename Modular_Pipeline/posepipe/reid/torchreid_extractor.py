"""
Torchreid backbone (OSNet / ResNet-50 / ...) as a ReID feature extractor.

Crops are taken from the detector box, resized to 256x128, ImageNet-normalised,
and the classifier head is removed so the pooled embedding is returned.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def crop_boxes(frame_bgr: np.ndarray, boxes_xyxy) -> list:
    H, W = frame_bgr.shape[:2]
    crops = []
    for b in np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4):
        x1, y1 = int(max(0, b[0])), int(max(0, b[1]))
        x2, y2 = int(min(W, b[2])), int(min(H, b[3]))
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
        crops.append(frame_bgr[y1:y2, x1:x2])
    return crops


class TorchreidExtractor:
    def __init__(self, model_name: str = "osnet_x1_0", weights: Optional[str] = None,
                 device: str = "cuda", num_classes: int = 1041, imagenet_init: bool = False,
                 batch_size: int = 64, **_):
        import torch
        import torch.nn as nn
        import torchreid
        import torchvision.transforms as T

        self.torch = torch
        self.device = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.model = torchreid.models.build_model(name=model_name, num_classes=num_classes,
                                                  pretrained=bool(imagenet_init))
        if weights and str(weights).lower() != "imagenet":
            self._load(Path(weights))
        else:
            log.warning("[ReID] %s without ReID weights (ImageNet features only).", model_name)
        if hasattr(self.model, "classifier"):
            self.model.classifier = nn.Identity()
        self.model.eval().to(self.device)
        self.batch_size = int(batch_size)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        log.info("[ReID] %s ready (device=%s)", model_name, self.device)

    def _load(self, w: Path) -> None:
        if not w.exists():
            raise FileNotFoundError(f"[ReID] weights not found: {w}")
        try:
            ckpt = self.torch.load(str(w), map_location="cpu")
        except UnicodeDecodeError:
            ckpt = self.torch.load(str(w), map_location="cpu", encoding="latin1")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace("module.", "").replace("backbone.", "").replace("base.", ""): v for k, v in sd.items()}
        own = self.model.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}  # drops mismatched heads
        missing, _ = self.model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("classifier")]
        log.info("[ReID] loaded %s (%d tensors, %d backbone tensors missing)", w.name, len(sd), len(missing))
        if missing:
            log.warning("[ReID] backbone tensors missing from checkpoint: %s ...", missing[:5])

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy) -> np.ndarray:
        torch = self.torch
        crops = crop_boxes(frame_bgr, boxes_xyxy)
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        feats = []
        with torch.no_grad():
            for s in range(0, len(crops), self.batch_size):
                x = torch.stack([self.transform(np.ascontiguousarray(c[:, :, ::-1]))
                                 for c in crops[s:s + self.batch_size]]).to(self.device)
                f = self.model(x)
                if isinstance(f, tuple):
                    f = f[0]
                feats.append(torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)
