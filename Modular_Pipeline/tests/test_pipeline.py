"""
End-to-end checks that need no GPU and no downloaded checkpoints.

Synthetic two-camera videos with two coloured "people" are run through the real
runner, trackers, stitching and merge code. Detector / pose / ReID are replaced
by tiny stand-ins registered into the same registries the real backends use.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from posepipe.config import apply_override, component_cfg, load_config, run_name  # noqa: E402
from posepipe.detection import DETECTORS  # noqa: E402
from posepipe.keypoints import convert  # noqa: E402
from posepipe.pose import POSE  # noqa: E402
from posepipe.reid import REID  # noqa: E402
from posepipe.runner import run_session  # noqa: E402

COLORS = {"red": (0, 0, 255), "green": (0, 255, 0)}
W, H, N_FRAMES = 320, 240, 40


# ----------------------------------------------------------------------------
# Stand-in components
# ----------------------------------------------------------------------------
@DETECTORS.register("fake_color")
class FakeColorDetector:
    """Boxes of pure red / green blobs."""

    def __init__(self, **_):
        pass

    def __call__(self, frame):
        out = []
        for bgr in COLORS.values():
            mask = np.all(frame == np.array(bgr, np.uint8), axis=2)
            ys, xs = np.nonzero(mask)
            if xs.size:
                out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9])
        return np.asarray(out, np.float32).reshape(-1, 5)


@POSE.register("fake_pose")
class FakePose:
    """17 joints on a grid inside the box, so keypoints identify their box."""
    num_joints = 17

    def __init__(self, **_):
        pass

    def __call__(self, frame, boxes):
        res = []
        for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
            t = np.linspace(0.1, 0.9, 17)
            kps = np.stack([x1 + t * (x2 - x1), y1 + t[::-1] * (y2 - y1), np.full(17, 0.9)], 1)
            res.append({"keypoints": kps.astype(np.float32), "score": 0.9})
        return res


@REID.register("fake_color_reid")
class FakeColorReID:
    """Embedding = normalised mean colour of the crop (+ small noise)."""

    def __init__(self, **_):
        self.rng = np.random.default_rng(0)

    def __call__(self, frame, boxes):
        feats = []
        for x1, y1, x2, y2 in np.asarray(boxes, int).reshape(-1, 4):
            f = frame[y1:y2, x1:x2].reshape(-1, 3).mean(0).astype(np.float32) + 1.0
            f = np.concatenate([f, self.rng.normal(0, 0.5, 5).astype(np.float32)])
            feats.append(f / np.linalg.norm(f))
        return np.stack(feats)


def make_video(path: Path, swap_lanes: bool = False):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for i in range(N_FRAMES):
        f = np.zeros((H, W, 3), np.uint8)
        # people walk towards each other and cross mid-way
        xr, xg = 20 + 5 * i, 260 - 5 * i
        yr, yg = (40, 140) if not swap_lanes else (140, 40)
        f[yr:yr + 60, xr:xr + 30] = COLORS["red"]
        f[yg:yg + 60, xg:xg + 30] = COLORS["green"]
        vw.write(f)
    vw.release()


def jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def base_cfg(tmp: Path, tracker: str) -> dict:
    cfg = load_config(overrides=[
        f"video_dir={tmp / 'videos'}", f"out_root={tmp / 'out'}",
        "device=cpu", "detector.type=fake_color", "pose.type=fake_pose",
        "reid.type=fake_color_reid", f"tracker.type={tracker}",
        "pipeline.min_track_length=5", "pipeline.kpt_thresh=0.5", "pipeline.min_mean_kp_conf=0.5",
        "pipeline.min_box_area_ratio=0.0", "pipeline.annotate_ids=true",
        "crosscam.global_merge.min_frames_per_gid=1", "crosscam.cos_thr=0.5",
    ])
    cfg["tracker"]["deepsort"]["n_init"] = 1
    return cfg


class _QuantizeDecoded:
    """Patch VideoReader frames so the colour detector sees exact colours."""

    def __enter__(self):
        import posepipe.io as pio
        self.pio, self.orig = pio, pio.VideoReader.__next__

        def nxt(reader):
            f = self.orig(reader)
            out = np.zeros_like(f)
            for bgr in COLORS.values():
                d = np.abs(f.astype(int) - np.array(bgr)).sum(2) < 120
                out[d] = bgr
            return out
        pio.VideoReader.__next__ = nxt
        return self

    def __exit__(self, *a):
        self.pio.VideoReader.__next__ = self.orig


# ----------------------------------------------------------------------------
class TestConfig(unittest.TestCase):
    def test_type_toggle_selects_block(self):
        cfg = load_config()
        self.assertEqual(component_cfg(cfg["pose"])["type"], "rtmpose")
        apply_override(cfg, "pose.type=alphapose")
        pc = component_cfg(cfg["pose"])
        self.assertTrue(pc["cfg"].endswith("256x192_w32_lr1e-3.yaml"))
        self.assertNotIn("input_color", pc)  # rtmpose-only key not leaked
        self.assertTrue(Path(pc["cfg"]).is_absolute())

    def test_overlay_and_run_name(self):
        cfg = load_config([str(ROOT / "configs/experiments/alphapose_stack.yaml")])
        self.assertEqual(run_name(cfg), "alphapose_osnet_strongsort")
        ss = component_cfg(cfg["tracker"])
        self.assertEqual(ss["max_cosine_distance"], 0.45)
        self.assertNotIn("n_init", ss)

    def test_keypoint_convert(self):
        wb = np.random.rand(133, 3).astype(np.float32)
        np.testing.assert_array_equal(convert(wb, "coco17"), wb[:17])
        up = convert(wb[:17], "coco_wholebody")
        self.assertEqual(up.shape, (133, 3))
        self.assertTrue(np.all(up[17:, 2] == 0))


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="posepipe_test_"))
        (cls.tmp / "videos").mkdir()
        make_video(cls.tmp / "videos" / "camA.mp4")
        make_video(cls.tmp / "videos" / "camB.mp4", swap_lanes=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, tracker):
        with _QuantizeDecoded():
            return run_session(base_cfg(self.tmp, tracker))

    def _check_run(self, out: Path):
        for cam in ("camA", "camB"):
            recs = jsonl(out / f"{cam}_output" / "results.jsonl")
            self.assertGreater(len(recs), 0)
            ids = {r["id"] for r in recs}
            self.assertEqual(len(ids), 2, f"{cam}: expected 2 identities, got {ids}")
            for r in recs:
                # keypoints must belong to the detection the track was matched to
                kps = np.asarray(r["keypoints"])
                x1, y1, x2, y2 = r["det_bbox"]
                self.assertTrue(np.all((kps[:, 0] >= x1) & (kps[:, 0] <= x2)), r)
            # each ID follows one person through the crossing (no swap)
            by_id = {}
            for r in recs:
                by_id.setdefault(r["id"], []).append(r)
            for tid, rs in by_id.items():
                first = min(rs, key=lambda r: r["frame"])
                last = max(rs, key=lambda r: r["frame"])
                moved = last["det_bbox"][0] - first["det_bbox"][0]
                self.assertGreater(abs(moved), 50, f"track {tid} did not follow one person")

        cmap = json.loads((out / "crosscam_map.json").read_text(encoding="utf-8"))
        for cam in ("camA_output", "camB_output"):
            local_ids = {str(r["id"]) for r in jsonl(out / cam / "results.jsonl")}
            self.assertEqual(set(cmap[cam]), local_ids, "map must be keyed by real local IDs")
        self.assertEqual(len({g for m in cmap.values() for g in m.values()}), 2)
        merged = jsonl(out / "merged_global_tracks_union.jsonl")
        self.assertTrue(merged and all("global_id" in r for r in merged))
        self.assertTrue((out / "camA_output" / "camA_output_globalids.mp4").exists())
        self.assertTrue((out / "run_config.yaml").exists())

    def test_deepsort(self):
        self._check_run(self._run("deepsort"))

    def test_strongsort(self):
        self._check_run(self._run("strongsort"))

    def test_evaluate(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")
        from posepipe.evaluation.metrics import run_eval
        out = self._run("strongsort")
        run_eval(str(out))
        self.assertTrue((out / "per_video.csv").exists())


class TestAlphaPoseBackend(unittest.TestCase):
    """Real AlphaPose code path with a randomly initialised HRNet (CPU)."""

    def test_forward_shapes(self):
        try:
            import torch
            import easydict  # noqa: F401
        except ImportError:
            self.skipTest("torch/easydict not installed")
        sys.path.insert(0, str(ROOT / "externals" / "AlphaPose"))
        from alphapose.models import builder
        from alphapose.utils.config import update_config
        cfgp = ROOT / "externals/AlphaPose/configs/coco/hrnet/256x192_w32_lr1e-3.yaml"
        c = update_config(str(cfgp))
        tmp = Path(tempfile.mkdtemp())
        ck = tmp / "rand.pth"
        torch.save(builder.build_sppe(c.MODEL, preset_cfg=c.DATA_PRESET).state_dict(), ck)

        est = POSE.build({"type": "alphapose", "cfg": str(cfgp), "ckpt": str(ck), "device": "cpu"})
        frame = np.random.randint(0, 255, (240, 320, 3), np.uint8)
        boxes = np.array([[10, 20, 60, 140], [200, 30, 260, 200]], np.float32)
        res = est(frame, boxes)
        self.assertEqual(len(res), 2)
        for r, (x1, y1, x2, y2) in zip(res, boxes):
            k = r["keypoints"]
            self.assertEqual(k.shape, (17, 3))
            # decoded joints land inside the aspect-expanded crop, in image coords
            w, h = x2 - x1, y2 - y1
            cw = max(w, h * 192 / 256) * 1.25
            ch = cw * 256 / 192
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.assertTrue(np.all(np.abs(k[:, 0] - cx) <= cw / 2 + 1), k[:, 0])
            self.assertTrue(np.all(np.abs(k[:, 1] - cy) <= ch / 2 + 1), k[:, 1])
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
