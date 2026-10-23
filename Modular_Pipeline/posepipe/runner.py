"""
Pipeline runner: detector -> pose -> ReID -> tracker per camera, then
cross-camera stitching, global-ID injection, merge and annotation.

Per-camera output: <out>/<video_stem>_output/{results.jsonl, annotated.mp4}
results.jsonl record (one per tracked person per frame):
  frame, id, bbox, det_bbox, keypoints (J,3), feature, score, pose_conf, det_score
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import component_cfg, dump_yaml, run_name
from .crosscam import annotate_videos_with_ids, inject_global_ids, merge_tracks_union, stitch
from .detection import build_detector
from .io import JsonlWriter, VideoReader, VideoWriter, read_jsonl
from .keypoints import convert, format_for, to_xyc
from .pose import build_pose
from .reid import build_reid
from .tracking import build_tracker
from .visualize import draw_pose_box_and_id

log = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# ----------------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------------
class Components:
    """Detector / pose / ReID models, built once and reused across videos."""

    def __init__(self, cfg: dict):
        device = cfg.get("device", "cuda")
        with_dev = lambda section: {"device": device, **component_cfg(section)}
        self.detector = build_detector(with_dev(cfg["detector"]))
        self.pose = build_pose(with_dev(cfg["pose"]))
        self.reid = build_reid(with_dev(cfg["reid"]))
        self.tracker_cfg = component_cfg(cfg["tracker"])

    def new_tracker(self):
        return build_tracker(self.tracker_cfg)


_CACHE: Dict[str, Components] = {}


def get_components(cfg: dict) -> Components:
    """Cache models by their config so parameter sweeps don't reload weights."""
    key = json.dumps({"device": cfg.get("device"),
                      **{k: component_cfg(cfg[k]) for k in ("detector", "pose", "reid")}},
                     sort_keys=True, default=str)
    if key not in _CACHE:
        _CACHE.clear()  # keep at most one set of models on the GPU
        _CACHE[key] = Components(cfg)
    comps = _CACHE[key]
    comps.tracker_cfg = component_cfg(cfg["tracker"])
    return comps


# ----------------------------------------------------------------------------
# Per-camera processing
# ----------------------------------------------------------------------------
def process_video(video_path, out_dir, cfg: dict, components: Optional[Components] = None) -> dict:
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comps = components or get_components(cfg)
    tracker = comps.new_tracker()
    pc = cfg.get("pipeline", {})

    kp_format = pc.get("keypoint_format", "native")
    conf_gate = max(float(pc.get("kpt_thresh", 0.0)), float(pc.get("min_mean_kp_conf", 0.0)))
    min_area = float(pc.get("min_box_area_ratio", 0.0))
    max_frames = int(pc.get("max_frames", -1))
    start_frame = int(pc.get("start_frame", 0))
    log_every = int(pc.get("log_every", 100))
    draw = bool(pc.get("draw", True)) and bool(pc.get("save_video", True))

    reader = VideoReader(video_path, start_frame=start_frame)
    writer = VideoWriter(out_dir / "annotated.mp4", reader.FPS, (reader.W, reader.H),
                         enabled=bool(pc.get("save_video", True)))
    jw = JsonlWriter(out_dir / "results.jsonl") if pc.get("save_json", True) else None

    print(f"[INFO] {video_path.name}: pose={cfg['pose']['type']} reid={cfg['reid']['type']} "
          f"tracker={cfg['tracker']['type']} frames={reader.frame_count or '?'} start={start_frame}")

    n, ids, t0 = 0, set(), time.time()
    frame_idx = start_frame
    for frame in reader:
        n += 1
        frame_idx += 1
        if 0 < max_frames < n:
            break

        dets = comps.detector(frame)
        if min_area > 0 and len(dets):
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            dets = dets[areas >= min_area * reader.W * reader.H]

        boxes, scores, kps_list = [], [], []
        if len(dets):
            for d, p in zip(dets, comps.pose(frame, dets[:, :4])):
                kps = convert(to_xyc(p["keypoints"]), kp_format)
                kps[:, 2] = np.clip(kps[:, 2], 0.0, 1.0)
                if float(kps[:, 2].mean()) < conf_gate:
                    continue
                boxes.append(d[:4])
                scores.append(float(d[4]))
                kps_list.append(kps)

        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        feats = comps.reid(frame, boxes_np) if len(boxes) else None
        tracks = tracker.update(boxes_np, scores, feats)

        for trk in tracks:
            i = trk["det_index"]
            kps = kps_list[i]
            pose_conf = float(kps[:, 2].mean())
            ids.add(trk["id"])
            if draw:
                draw_pose_box_and_id(frame, kps, trk["id"], bbox=trk["bbox"])
            if jw:
                jw.write({
                    "frame": int(frame_idx),
                    "id": int(trk["id"]),
                    "bbox": [float(x) for x in trk["bbox"]],
                    "det_bbox": [float(x) for x in boxes_np[i]],
                    "keypoints": kps.tolist(),
                    "feature": feats[i].tolist() if feats is not None else [],
                    "score": pose_conf,
                    "pose_conf": pose_conf,
                    "det_score": scores[i],
                })
        writer.write(frame)
        if n % log_every == 0:
            print(f"[INFO] {video_path.stem}: frame {frame_idx} ({n / (time.time() - t0):.1f} fps)")

    reader.close()
    writer.close()
    if jw:
        jw.close()
        filter_short_tracks(out_dir / "results.jsonl", int(pc.get("min_track_length", 1)))

    return {
        "video_name": video_path.stem,
        "frames": n,
        "fps": reader.FPS,
        "tracks": len(ids),
        "pose": cfg["pose"]["type"],
        "keypoint_format": kp_format if kp_format != "native" else format_for(getattr(comps.pose, "num_joints", 0)),
        "reid": cfg["reid"]["type"],
        "tracker": cfg["tracker"]["type"],
        "seconds": round(time.time() - t0, 1),
        "output_dir": str(out_dir),
    }


def filter_short_tracks(jsonl_path: Path, min_track_len: int) -> None:
    """Drop local tracks shorter than min_track_len frames (in place)."""
    if min_track_len <= 1 or not jsonl_path.exists():
        return
    counts: Dict[int, int] = {}
    for item in read_jsonl(jsonl_path):
        counts[item["id"]] = counts.get(item["id"], 0) + 1
    keep = {t for t, c in counts.items() if c >= min_track_len}
    tmp = jsonl_path.with_suffix(".tmp.jsonl")
    kept = total = 0
    with open(tmp, "w", encoding="utf-8") as fw:
        for item in read_jsonl(jsonl_path):
            total += 1
            if item["id"] in keep:
                fw.write(json.dumps(item) + "\n")
                kept += 1
    os.replace(tmp, jsonl_path)
    print(f"[Filter] {jsonl_path.parent.name}: kept {len(keep)}/{len(counts)} tracks "
          f">= {min_track_len} frames ({kept}/{total} records)")


# ----------------------------------------------------------------------------
# Multi-camera session
# ----------------------------------------------------------------------------
def list_videos(cfg: dict) -> List[Path]:
    if cfg.get("videos"):
        return [Path(v) for v in cfg["videos"]]
    vd = Path(cfg.get("video_dir", "."))
    if vd.is_file():
        return [vd]
    return sorted(p for p in vd.iterdir() if p.suffix.lower() in VIDEO_EXTS) if vd.is_dir() else []


def run_crosscam(out_root: Path, cfg: dict, videos: List[Path]) -> None:
    cc = cfg.get("crosscam", {})
    gm = cc.get("global_merge", {})
    im = cc.get("intra_merge", {})
    cmap = stitch(out_root, cos_thr=float(cc.get("cos_thr", 0.4)), alpha=float(cc.get("alpha", 0.5)),
                  intra_emb_thr=float(im.get("emb_thr", 0.15)), intra_time_gap=int(im.get("time_gap", 30)),
                  intra_iou_thr=float(im.get("iou_thr", 0.0)), min_frames=int(cc.get("min_frames", 3)))
    if not cmap:
        print("[WARN] empty cross-camera map; skipping merge/annotation.")
        return
    inject_global_ids(out_root, cmap)
    merge_tracks_union(out_root, cmap, out_root / "merged_global_tracks_union.jsonl",
                       min_frames_per_gid=int(gm.get("min_frames_per_gid", 80)),
                       require_multi_cam=bool(gm.get("require_multi_cam", False)))
    if cfg.get("pipeline", {}).get("annotate_ids", False):
        annotate_videos_with_ids(out_root, {v.stem: v for v in videos},
                                 start_frame=int(cfg.get("pipeline", {}).get("start_frame", 0)))


def run_session(cfg: dict, out_root: Optional[Path] = None) -> Path:
    videos = list_videos(cfg)
    if not videos:
        raise SystemExit(f"No videos found (video_dir={cfg.get('video_dir')!r}, videos={cfg.get('videos')!r})")

    if out_root is None:
        out_root = Path(cfg.get("out_root", "outputs")) / run_name(cfg)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, out_root / "run_config.yaml")  # exact config used, for provenance
    print(f"[INFO] {len(videos)} video(s) -> {out_root}")

    comps = get_components(cfg)
    summaries = [process_video(v, out_root / f"{v.stem}_output", cfg, comps) for v in videos]
    with open(out_root / "summary_videos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    if cfg.get("crosscam", {}).get("enabled", True):
        run_crosscam(out_root, cfg, videos)
    print(f"[DONE] {out_root}")
    return out_root
