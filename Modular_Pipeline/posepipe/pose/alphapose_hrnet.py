"""
AlphaPose SPPE (HRNet-W32 256x192 by default) as a top-down backend.

Preprocessing and decoding follow AlphaPose's own inference path
(SimpleTransform.test_transform + heatmap_to_coord_simple):
  - aspect-ratio preserving box expansion (x1.25)
  - RGB in [0,1] minus mean (0.406, 0.457, 0.480), *no* std division
  - argmax + quarter-pixel refinement, mapped back with transform_preds

Only pure-PyTorch parts of AlphaPose are imported, so no `setup.py build`
(CUDA extensions) is needed for HRNet / FastPose models.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from ..config import PROJECT_ROOT

log = logging.getLogger(__name__)

DEFAULT_ROOT = PROJECT_ROOT / "externals" / "AlphaPose"


class AlphaPoseEstimator:
    def __init__(self, cfg: str, ckpt: str, device: str = "cuda",
                 alphapose_root: Optional[str] = None, batch_size: int = 64,
                 fp16: bool = False, **_):
        root = Path(alphapose_root or DEFAULT_ROOT).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from alphapose.models import builder
        from alphapose.utils.bbox import _box_to_center_scale, _center_scale_to_box
        from alphapose.utils.config import update_config
        from alphapose.utils.transforms import get_affine_transform, heatmap_to_coord_simple

        self._box_to_center_scale = _box_to_center_scale
        self._center_scale_to_box = _center_scale_to_box
        self._get_affine_transform = get_affine_transform
        self._heatmap_to_coord = heatmap_to_coord_simple

        cfg_path, ckpt_path = Path(cfg), Path(ckpt)
        for p in (cfg_path, ckpt_path):
            if not p.is_file():
                raise FileNotFoundError(f"[AlphaPose] file not found: {p}")

        ap_cfg = update_config(str(cfg_path))
        self.in_h, self.in_w = [int(v) for v in ap_cfg.DATA_PRESET.IMAGE_SIZE]
        self.aspect_ratio = float(self.in_w) / float(self.in_h)
        self.num_joints = int(ap_cfg.DATA_PRESET.NUM_JOINTS)

        self.model = builder.build_sppe(ap_cfg.MODEL, preset_cfg=ap_cfg.DATA_PRESET)
        self._load_weights(ckpt_path)

        use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(device if use_cuda else "cpu")
        self.fp16 = bool(fp16 and use_cuda)
        self.model.to(self.device).eval()
        if self.fp16:
            self.model.half()
        self.batch_size = int(batch_size)
        log.info("AlphaPose SPPE ready: %s (%dx%d, %d joints, device=%s)",
                 ap_cfg.MODEL.TYPE, self.in_h, self.in_w, self.num_joints, self.device)

    # ------------------------------------------------------------------
    def _load_weights(self, ckpt_path: Path) -> None:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        total = len(self.model.state_dict())
        log.info("[AlphaPose] weights: %d/%d tensors loaded (missing=%d unexpected=%d)",
                 total - len(missing), total, len(missing), len(unexpected))
        if len(missing) > 0.4 * total:
            raise RuntimeError(
                f"[AlphaPose] only {total - len(missing)}/{total} tensors matched {ckpt_path.name}. "
                "Check the cfg/ckpt pairing (e.g. w32 vs w48, 256x192 vs 384x288).")

    def _crop(self, img_rgb: np.ndarray, box):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        center, scale = self._box_to_center_scale(x1, y1, x2 - x1, y2 - y1, self.aspect_ratio)
        trans = self._get_affine_transform(center, scale, 0, [self.in_w, self.in_h])
        crop = cv2.warpAffine(img_rgb, trans, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR)
        ten = torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32) / 255.0)
        ten[0].add_(-0.406)
        ten[1].add_(-0.457)
        ten[2].add_(-0.480)
        return ten, self._center_scale_to_box(center, scale)

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy) -> List[dict]:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4) if len(boxes_xyxy) else np.empty((0, 4))
        if boxes.shape[0] == 0:
            return []
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crops, crop_boxes = zip(*(self._crop(img_rgb, b) for b in boxes))

        out: List[dict] = []
        for s in range(0, len(crops), self.batch_size):
            inp = torch.stack(crops[s:s + self.batch_size]).to(self.device)
            if self.fp16:
                inp = inp.half()
            hm = self.model(inp)
            if isinstance(hm, (list, tuple)):
                hm = hm[-1]
            hm = hm.float().cpu().numpy()
            for i in range(hm.shape[0]):
                coords, maxvals = self._heatmap_to_coord(hm[i], crop_boxes[s + i])
                conf = np.asarray(maxvals, dtype=np.float32).reshape(-1)
                kps = np.concatenate([np.asarray(coords, np.float32).reshape(-1, 2), conf[:, None]], axis=1)
                out.append({"keypoints": kps, "score": float(conf.mean())})
        return out
