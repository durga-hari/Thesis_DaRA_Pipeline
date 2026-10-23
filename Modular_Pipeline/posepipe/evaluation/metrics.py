#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised metrics for one run of the modular pipeline (any pose backend).
Ported from RTMO_DeepSORT_Pipeline/Pipeline/rtmo_evaluation_pipeline_metrics.py;
the metric definitions are unchanged.

Outputs in the run root:
  - per_track.csv           : per-track metrics (track_len, proxy switch counts)
  - per_video.csv           : per-video metrics (detector, pose, tracking, ReID, yields)
  - overall.csv             : aggregated metrics over all videos
  - limb_lengths.csv        : per-frame bone lengths for all tracks
  - jitter_timeseries.csv   : per-frame per-joint jitter for all tracks (COCO-WB: 133 joints)
  - per_frame_metrics.csv   : per-frame detector & tracking metrics
  - crosscam.csv            : cross-camera scalar metrics (if global JSONL present)

Assumes inputs:
  - A run root written by `posepipe run` (<out_root>/<run_name>)
  - One or more *_poses_union.jsonl files under that root
  - Optional merged_global_tracks_union.jsonl for cross-cam metrics

Keypoints:
  - The joint count K is inferred from the data (17 COCO or 133 WholeBody).
  - Jitter is computed for all K joints.
  - Limb metrics use COCO body limbs (indices < 17), valid for both formats,
    so limb / smoothness metrics are directly comparable across backends.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# Set by run_eval(): joint count of the run and the label written to "mode".
NUM_JOINTS = 133
MODE = "pipeline"
DEFAULT_GLOBAL_FILE = "merged_global_tracks_union.jsonl"

IOU_CONT_THR = 0.3
EMBED_JUMP_THR = 0.7

# Major limbs + shoulders/hips (COCO body indices inside COCO-WholeBody)
BONES_DEFAULT = [
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (5, 7),  (7, 9),     # left arm
    (6, 8),  (8, 10),    # right arm
    (5, 6), (11, 12),    # shoulders / hips
]

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    IoU between [x1,y1,x2,y2] boxes.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0)
    union = max(area_a + area_b - inter, 1e-9)
    return float(inter / union)


def flatten_keypoints_xy(keypoints: np.ndarray) -> np.ndarray:
    """
    keypoints: [J, 2 or 3] -> [2J] (x1,y1,x2,y2,...)
    """
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("keypoints must be [J,>=2]")
    return keypoints[:, :2].reshape(-1)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# RTMO JSON/JSONL → frame_records[]
# ---------------------------------------------------------------------------

def rtmo_maybe_flatten_kps(kps):
    """
    Accept [x,y,score]*K or [[x,y,score], ...] or [[x,y], ...]
    → flat [x,y,score,...] with score=1.0 if missing.
    """
    if kps is None:
        return None
    if len(kps) == 0:
        return []
    if isinstance(kps[0], (list, tuple, np.ndarray)):
        flat = []
        for tri in kps:
            if len(tri) == 3:
                flat.extend(tri)
            elif len(tri) == 2:
                flat.extend([tri[0], tri[1], 1.0])
        return flat
    return kps


def rtmo_keypoints_to_arr(kps_flat, k: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Convert flat [x,y,score,...] to [K,3] (x,y,score), padding/truncating to K joints.

    Here we set K=133 to match COCO-WholeBody. If fewer than 133 joints are
    present they are padded with NaNs; if more, they are truncated.
    """
    if kps_flat is None:
        return None
    k = NUM_JOINTS if k is None else k
    arr = np.asarray(kps_flat, dtype=np.float32)
    if arr.size == 0 or arr.size % 3 != 0:
        return None
    arr = arr.reshape(-1, 3)
    if arr.shape[0] < k:
        pad = np.full((k - arr.shape[0], 3), np.nan, dtype=np.float32)
        arr = np.vstack([arr, pad])
    elif arr.shape[0] > k:
        arr = arr[:k]
    return arr


def parse_rtmo_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize RTMO detection record to:
      { frame_idx, track_id, bbox, score, keypoints, embedding }
    """
    try:
        frame = int(
            raw.get("frame_id", raw.get("frame", raw.get("frameIndex", -1)))
        )
        tid = raw.get("id", raw.get("track_id", raw.get("tid", None)))
        bbox = raw.get("bbox", raw.get("box", None))
        score = raw.get("conf", raw.get("score", raw.get("confidence", None)))
        feat = raw.get("feature", raw.get("reid", None))
        kps = raw.get("kps", raw.get("keypoints", raw.get("pose", None)))

        if tid is None or bbox is None or frame < 0:
            return None

        bbox = [float(b) for b in bbox]
        if len(bbox) != 4:
            return None

        feat = None if feat is None else [float(x) for x in feat]
        kps = rtmo_maybe_flatten_kps(kps)
        kps_arr = rtmo_keypoints_to_arr(kps) if kps is not None else None

        return dict(
            frame_idx=frame,
            track_id=str(tid),
            bbox=np.asarray(bbox, dtype=float),
            score=float(score) if score is not None else float("nan"),
            keypoints=kps_arr,
            embedding=np.asarray(feat, dtype=float) if feat is not None else None,
        )
    except Exception:
        return None


def load_rtmo_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load RTMO *_poses_union.jsonl or plain JSON into normalized detection records.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")
    items: List[Dict[str, Any]] = []

    # Try JSONL
    if "\n" in txt:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            rec = parse_rtmo_record(raw)
            if rec:
                items.append(rec)
        if items:
            return items

    # Fallback: plain JSON (list/dict)
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            for raw in data:
                rec = parse_rtmo_record(raw)
                if rec:
                    items.append(rec)
        elif isinstance(data, dict):
            seq = data.get("frames", data.get("data", []))
            for raw in seq:
                rec = parse_rtmo_record(raw)
                if rec:
                    items.append(rec)
    except Exception:
        pass

    return items


def build_frame_records_rtmo(path: Path) -> List[Dict[str, Any]]:
    """
    RTMO records → frame_records for downstream metrics.

    frame_records:
      [
        {
          "frame_idx": int,
          "cam_id": str,
          "detections": [
            {
              "track_id": ...,
              "bbox": np.array([x1,y1,x2,y2]),
              "score": float,
              "keypoints": np.ndarray[133,3] or None,
              "embedding": np.ndarray[D] or None,
            }, ...
          ]
        }, ...
      ]
    """
    # infer camera ID from filename/parent: e.g. AC11_output_poses_union.jsonl → "AC11"
    fname = path.name
    cam_id = fname.split("_")[0] if "_" in fname else path.parent.name.split("_")[0]

    dets = load_rtmo_records(path)
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in dets:
        by_frame[d["frame_idx"]].append({
            "track_id": d["track_id"],
            "bbox": d["bbox"],
            "score": d["score"],
            "keypoints": d["keypoints"],
            "embedding": d["embedding"],
        })

    frames: List[Dict[str, Any]] = []
    for f in sorted(by_frame.keys()):
        frames.append({
            "frame_idx": int(f),
            "cam_id": cam_id,
            "detections": by_frame[f],
        })
    return frames


# ---------------------------------------------------------------------------
# Tracks from frame_records
# ---------------------------------------------------------------------------

def build_tracks_from_frame_records(
    frame_records: List[Dict[str, Any]]
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    track_id -> list of entries sorted by frame_idx.
    each entry: {frame_idx, cam_id, bbox, keypoints, embedding, score}
    """
    tracks: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for fr in frame_records:
        fidx = fr["frame_idx"]
        cam_id = fr.get("cam_id")
        for det in fr.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            tracks[tid].append({
                "frame_idx": fidx,
                "cam_id": cam_id,
                "bbox": det.get("bbox"),
                "keypoints": det.get("keypoints"),
                "embedding": det.get("embedding"),
                "score": det.get("score", float("nan")),
            })

    for tid in tracks:
        tracks[tid].sort(key=lambda e: e["frame_idx"])
    return tracks


# ---------------------------------------------------------------------------
# Per-track metrics
# ---------------------------------------------------------------------------

def compute_proxy_id_switches(
    entries: List[Dict[str, Any]],
    iou_thr: float = IOU_CONT_THR,
    embed_jump_thr: float = EMBED_JUMP_THR,
) -> Tuple[int, int]:
    """
    Proxy for ID switch inside a single track:
    count events where temporal IoU is low AND embedding jump is large.
    """
    if len(entries) < 2:
        return 0, 0
    switches = 0
    checks = 0
    for i in range(len(entries) - 1):
        b1 = entries[i].get("bbox")
        b2 = entries[i + 1].get("bbox")
        if b1 is None or b2 is None:
            continue
        iou = bbox_iou(np.asarray(b1, float), np.asarray(b2, float))
        f1 = entries[i].get("embedding")
        f2 = entries[i + 1].get("embedding")
        if f1 is None or f2 is None:
            continue
        f1 = np.asarray(f1, dtype=np.float32)
        f2 = np.asarray(f2, dtype=np.float32)
        if f1.shape != f2.shape:
            continue
        dj = float(np.linalg.norm(f1 - f2))
        checks += 1
        if iou < iou_thr and dj > embed_jump_thr:
            switches += 1
    return switches, checks


def compute_track_length(entries: List[Dict[str, Any]]) -> int:
    return len(entries)


# ---------------------------------------------------------------------------
# Limb length & jitter time series
# ---------------------------------------------------------------------------

def compute_limb_length_timeseries(
    tracks: Dict[Any, List[Dict[str, Any]]],
    bones: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    if bones is None:
        bones = BONES_DEFAULT

    rows = []
    for tid, entries in tracks.items():
        for e in entries:
            kps = e.get("keypoints")
            if kps is None or kps.size == 0:
                continue
            xy = kps[:, :2]
            J = xy.shape[0]
            for (i, j) in bones:
                if i >= J or j >= J:
                    continue
                p1, p2 = xy[i], xy[j]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue
                length = float(np.linalg.norm(p1 - p2))
                rows.append({
                    "track_id": tid,
                    "frame_idx": e["frame_idx"],
                    "cam_id": e.get("cam_id"),
                    "bone": f"{i}-{j}",
                    "joint_i": i,
                    "joint_j": j,
                    "limb_length": length,
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "track_id", "frame_idx", "cam_id",
                "bone", "joint_i", "joint_j", "limb_length"
            ]
        )
    return pd.DataFrame(rows)


def compute_jitter_timeseries(
    tracks: Dict[Any, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Per-frame per-joint jitter magnitudes (L2 displacement) for each track.

    For COCO-WholeBody we will have 133 joints; this will emit 133 rows
    (joint_idx = 0..132) per frame per track.
    """
    rows = []
    for tid, entries in tracks.items():
        if len(entries) < 2:
            continue
        for i in range(1, len(entries)):
            prev = entries[i - 1]
            cur = entries[i]
            k1 = prev.get("keypoints")
            k2 = cur.get("keypoints")
            if k1 is None or k2 is None:
                continue
            if k1.shape != k2.shape:
                continue
            dv = k2[:, :2] - k1[:, :2]
            mag = np.linalg.norm(dv, axis=1)
            for jid, val in enumerate(mag):
                rows.append({
                    "track_id": tid,
                    "frame_idx": cur["frame_idx"],
                    "cam_id": cur.get("cam_id"),
                    "joint_idx": jid,
                    "jitter_value": float(val),
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "track_id", "frame_idx", "cam_id",
                "joint_idx", "jitter_value"
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection metrics (per-video)
# ---------------------------------------------------------------------------

def detection_frame_stats(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = [len(fr.get("detections", [])) for fr in frame_records]
    mean_c = safe_mean(counts)
    std_c = safe_std(counts)
    fds = float(std_c / (mean_c + 1e-9)) if np.isfinite(mean_c) else float("nan")
    return {
        "mean_dets_per_frame": mean_c,
        "frame_detection_stability": fds,
    }


def detection_temporal_persistence(
    frame_records: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    frames_sorted = sorted(frame_records, key=lambda fr: fr["frame_idx"])
    if len(frames_sorted) < 2:
        return {"temporal_persistence_rate": float("nan")}

    num_dets = 0
    num_persist = 0
    for i in range(len(frames_sorted) - 1):
        fr = frames_sorted[i]
        fr_next = frames_sorted[i + 1]
        dets = fr.get("detections", [])
        dets_next = fr_next.get("detections", [])
        boxes_next = [d["bbox"] for d in dets_next if d.get("bbox") is not None]
        for det in dets:
            box = det.get("bbox")
            if box is None or not boxes_next:
                continue
            num_dets += 1
            best_iou = max(bbox_iou(box, b) for b in boxes_next)
            if best_iou >= iou_thresh:
                num_persist += 1
    tpr = float(num_persist / num_dets) if num_dets > 0 else float("nan")
    return {"temporal_persistence_rate": tpr}


def detection_confidence_stats(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    scores = []
    for fr in frame_records:
        for det in fr.get("detections", []):
            s = det.get("score")
            if s is not None:
                scores.append(float(s))
    return {
        "mean_detection_score": safe_mean(scores),
        "std_detection_score": safe_std(scores),
    }


# ---------------------------------------------------------------------------
# Pose metrics (per-video, scalar)
# ---------------------------------------------------------------------------

def pose_confidence_stats(tracks: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, float]:
    confs = []
    for _, entries in tracks.items():
        for e in entries:
            kps = e.get("keypoints")
            if kps is None or kps.size == 0:
                continue
            if kps.shape[1] >= 3:
                c = kps[:, 2]
            else:
                c = np.ones(kps.shape[0], dtype=float)
            confs.extend(c.tolist())
    return {
        "mean_pose_confidence": safe_mean(confs),
        "std_pose_confidence": safe_std(confs),
    }


def pose_limb_length_consistency(
    tracks: Dict[Any, List[Dict[str, Any]]],
    bones: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    if bones is None:
        bones = BONES_DEFAULT

    cvs = []
    for _, entries in tracks.items():
        limb_len_per_bone = {b: [] for b in bones}
        for e in entries:
            kps = e.get("keypoints")
            if kps is None:
                continue
            xy = kps[:, :2]
            J = xy.shape[0]
            for b in bones:
                i, j = b
                if i >= J or j >= J:
                    continue
                p1, p2 = xy[i], xy[j]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue
                l = float(np.linalg.norm(p1 - p2))
                limb_len_per_bone[b].append(l)

        per_track_cvs = []
        for vals in limb_len_per_bone.values():
            if len(vals) < 2:
                continue
            arr = np.asarray(vals, dtype=float)
            m = np.mean(arr)
            s = np.std(arr)
            if m > 1e-6:
                per_track_cvs.append(float(s / m))
        if per_track_cvs:
            cvs.append(float(np.mean(per_track_cvs)))

    return {
        "mean_limb_length_cv_over_tracks": safe_mean(cvs),
        "std_limb_length_cv_over_tracks": safe_std(cvs),
    }


def pose_temporal_smoothness(tracks: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Mean/STD of average per-joint displacement between consecutive frames
    (called "pose jitter" in AP script).
    """
    vels = []
    for _, entries in tracks.items():
        if len(entries) < 2:
            continue
        for i in range(1, len(entries)):
            k1 = entries[i - 1].get("keypoints")
            k2 = entries[i].get("keypoints")
            if k1 is None or k2 is None:
                continue
            if k1.shape != k2.shape:
                continue
            dv = k2[:, :2] - k1[:, :2]
            mag = np.linalg.norm(dv, axis=1)
            if np.isnan(mag).all():
                continue
            vels.append(float(np.nanmean(mag)))
    return {
        "mean_pose_jitter": safe_mean(vels),
        "std_pose_jitter": safe_std(vels),
    }


def pose_temporal_cosine(tracks: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Cosine similarity of flattened poses between consecutive frames
    (self-similarity of skeletons over time).
    """
    sims = []
    for _, entries in tracks.items():
        if len(entries) < 2:
            continue
        prev_vec = None
        for e in entries:
            kps = e.get("keypoints")
            if kps is None:
                prev_vec = None
                continue
            vec = flatten_keypoints_xy(kps)
            if prev_vec is not None and vec.shape == prev_vec.shape:
                num = float(np.dot(vec, prev_vec))
                den = float(np.linalg.norm(vec) * np.linalg.norm(prev_vec) + 1e-9)
                sims.append(num / den)
            prev_vec = vec
    return {
        "mean_pose_temporal_cosine": safe_mean(sims),
        "std_pose_temporal_cosine": safe_std(sims),
    }


# ---------------------------------------------------------------------------
# Tracking metrics (per-video)
# ---------------------------------------------------------------------------

def tracking_basic_stats(
    tracks: Dict[Any, List[Dict[str, Any]]],
    total_frames: int,
) -> Dict[str, float]:
    lengths = [len(v) for v in tracks.values()]
    if not lengths:
        return {
            "num_tracks": 0.0,
            "mean_track_length": float("nan"),
            "track_fragmentation_index": float("nan"),
            "track_continuity_ratio": float("nan"),
        }
    mean_len = safe_mean(lengths)
    min_len = 10
    frac_short = float(sum(1 for L in lengths if L < min_len) / len(lengths))
    tcr = float(mean_len / (total_frames + 1e-9)) if total_frames > 0 else float("nan")
    return {
        "num_tracks": float(len(lengths)),
        "mean_track_length": mean_len,
        "track_fragmentation_index": frac_short,
        "track_continuity_ratio": tcr,
    }


# ---------------------------------------------------------------------------
# ReID metrics (per-video)
# ---------------------------------------------------------------------------

def reid_temporal_embedding_std(
    tracks: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, float]:
    track_stds = []
    for _, entries in tracks.items():
        embs = [e.get("embedding") for e in entries]
        embs = [np.asarray(x, dtype=float) for x in embs if x is not None]
        if len(embs) < 2:
            continue
        X = np.stack(embs, axis=0)
        per_dim_std = np.std(X, axis=0)
        track_stds.append(float(np.mean(per_dim_std)))
    return {
        "mean_embedding_temporal_std": safe_mean(track_stds),
        "std_embedding_temporal_std": safe_std(track_stds),
    }


def reid_embedding_dispersion(
    tracks: Dict[Any, List[Dict[str, Any]]],
    max_samples: int = 3000,
) -> Dict[str, float]:
    """
    Global embedding dispersion:
    mean/STD of cosine distance between random embedding pairs.
    """
    embs = []
    for _, entries in tracks.items():
        for e in entries:
            emb = e.get("embedding")
            if emb is None:
                continue
            embs.append(np.asarray(emb, dtype=float))
            if len(embs) >= max_samples:
                break
        if len(embs) >= max_samples:
            break
    if len(embs) < 2:
        return {
            "embedding_dispersion_mean": float("nan"),
            "embedding_dispersion_std": float("nan"),
        }
    X = np.stack(embs, axis=0)
    n = X.shape[0]
    N_pairs = min(20000, n * (n - 1) // 2)
    idx_i = np.random.randint(0, n, size=N_pairs)
    idx_j = np.random.randint(0, n, size=N_pairs)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    a = X[idx_i]
    b = X[idx_j]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    cos_sim = num / den
    cos_dist = 1.0 - cos_sim
    return {
        "embedding_dispersion_mean": float(np.mean(cos_dist)),
        "embedding_dispersion_std": float(np.std(cos_dist)),
    }


# ---------------------------------------------------------------------------
# Cross-camera metrics (scalar)
# ---------------------------------------------------------------------------

def build_global_tracks(global_path: Path) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Supports both AP-style and flattened RTMO-style global JSONL:

      A) per-frame with "poses" + "embeds"
      B) flattened per-detection with "global_id" fields
    """
    tracks: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in load_jsonl(global_path):
        if "poses" in rec:
            # AP-style bundle
            cam_id = rec.get("cam_id")
            frame_idx = rec.get("frame") or rec.get("frame_idx")
            poses = rec.get("poses") or []
            embeds = rec.get("embeds") or []
            for i, person in enumerate(poses):
                gid = person.get("global_id") or person.get("gid")
                if gid is None:
                    continue
                emb = embeds[i] if i < len(embeds) else None
                kps = (
                    person.get("keypoints")
                    or person.get("pose")
                    or person.get("joints")
                    or person.get("kps")
                )
                bbox = person.get("bbox") or person.get("box")
                tracks[gid].append({
                    "cam_id": cam_id,
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "keypoints": kps,
                    "embedding": emb,
                })
        else:
            # flattened RTMO-style
            gid = rec.get("global_id") or rec.get("gid")
            if gid is None:
                continue
            cam_id = rec.get("cam_id")
            frame_idx = rec.get("frame") or rec.get("frame_idx")
            bbox = rec.get("bbox") or rec.get("box")
            kps = (
                rec.get("keypoints")
                or rec.get("pose")
                or rec.get("joints")
                or rec.get("kps")
            )
            emb = (
                rec.get("embedding")
                or rec.get("embed")
                or rec.get("feature")
                or rec.get("reid")
            )
            tracks[gid].append({
                "cam_id": cam_id,
                "frame_idx": frame_idx,
                "bbox": bbox,
                "keypoints": kps,
                "embedding": emb,
            })
    return tracks


def crosscam_graph_connectivity(global_tracks: Dict[Any, List[Dict[str, Any]]]
                                ) -> Dict[str, float]:
    if not global_tracks:
        return {
            "global_id_count": 0.0,
            "graph_connectivity_ratio": float("nan"),
        }
    multi_cam = 0
    for _, entries in global_tracks.items():
        cams = {e.get("cam_id") for e in entries if e.get("cam_id") is not None}
        if len(cams) > 1:
            multi_cam += 1
    total = len(global_tracks)
    gcr = float(multi_cam / total) if total > 0 else float("nan")
    return {
        "global_id_count": float(total),
        "graph_connectivity_ratio": gcr,
    }


def crosscam_pairwise_embedding_consistency(
    global_tracks: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, float]:
    sims_per_gid = []
    for _, entries in global_tracks.items():
        per_cam = defaultdict(list)
        for e in entries:
            cam = e.get("cam_id")
            emb = e.get("embedding")
            if cam is None or emb is None:
                continue
            per_cam[cam].append(np.asarray(emb, dtype=float))
        cams = list(per_cam.keys())
        if len(cams) < 2:
            continue
        cam_mean = {}
        for c in cams:
            X = np.stack(per_cam[c], axis=0)
            cam_mean[c] = np.mean(X, axis=0)
        cs = []
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                a = cam_mean[cams[i]]
                b = cam_mean[cams[j]]
                num = float(np.dot(a, b))
                den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                cs.append(num / den)
        if cs:
            sims_per_gid.append(float(np.mean(cs)))
    return {
        "mean_crosscam_embedding_similarity": safe_mean(sims_per_gid),
        "std_crosscam_embedding_similarity": safe_std(sims_per_gid),
    }


def compute_crosscam_metrics(global_path: Path) -> Dict[str, float]:
    if not global_path.exists():
        return {}
    global_tracks = build_global_tracks(global_path)
    if not global_tracks:
        return {}
    metrics: Dict[str, float] = {}
    metrics.update(crosscam_graph_connectivity(global_tracks))
    metrics.update(crosscam_pairwise_embedding_consistency(global_tracks))
    return metrics


# ---------------------------------------------------------------------------
# Pipeline health (per-video scalar yields)
# ---------------------------------------------------------------------------

def pipeline_stage_yield(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    total = 0
    with_pose = 0
    with_emb = 0
    for fr in frame_records:
        for det in fr.get("detections", []):
            total += 1
            if det.get("keypoints") is not None:
                with_pose += 1
            if det.get("embedding") is not None:
                with_emb += 1
    if total == 0:
        return {
            "total_detections": 0.0,
            "yield_pose": float("nan"),
            "yield_embedding": float("nan"),
        }
    return {
        "total_detections": float(total),
        "yield_pose": float(with_pose / total),
        "yield_embedding": float(with_emb / total),
    }


# ---------------------------------------------------------------------------
# Per-frame metrics (detector + tracking)
# ---------------------------------------------------------------------------

def compute_per_frame_metrics(
    frame_records: List[Dict[str, Any]],
    cam_id: str,
    mode: str,
) -> pd.DataFrame:
    rows = []
    for fr in frame_records:
        fidx = fr["frame_idx"]
        dets = fr.get("detections", [])
        n_dets = len(dets)
        scores = [d.get("score") for d in dets if d.get("score") is not None]
        mean_det_score = safe_mean(scores) if scores else float("nan")
        tids = {d.get("track_id") for d in dets if d.get("track_id") is not None}
        n_tracks_active = len(tids)
        rows.append({
            "cam_id": cam_id,
            "frame_idx": int(fidx),
            "mode": mode,
            "n_dets": int(n_dets),
            "mean_det_score": float(mean_det_score),
            "n_tracks_active": int(n_tracks_active),
        })
    if not rows:
        return pd.DataFrame(
            columns=[
                "cam_id", "frame_idx", "mode",
                "n_dets", "mean_det_score", "n_tracks_active"
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core: compute metrics for one file
# ---------------------------------------------------------------------------

def compute_all_metrics_for_file(
    frame_records: List[Dict[str, Any]],
    cam_id: str,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      per_track_df    : per-track metrics
      limb_df         : per-frame limb lengths
      jitter_df       : per-frame per-joint jitter
      per_frame_df    : per-frame detector & tracking metrics
      per_video_row   : dict with aggregated metrics for this file
    """
    tracks = build_tracks_from_frame_records(frame_records)
    total_frames = len(frame_records)

    # per-track metrics (track length, proxy switches)
    per_track_rows = []
    total_proxy_sw = 0
    total_proxy_chk = 0
    for tid, entries in tracks.items():
        track_len = compute_track_length(entries)
        sw, chk = compute_proxy_id_switches(entries)
        total_proxy_sw += sw
        total_proxy_chk += chk
        per_track_rows.append({
            "track_id": tid,
            "cam_id": cam_id,
            "mode": mode,
            "track_len": int(track_len),
        })
    per_track_df = pd.DataFrame(per_track_rows)

    limb_df = compute_limb_length_timeseries(tracks)
    jitter_df = compute_jitter_timeseries(tracks)
    per_frame_df = compute_per_frame_metrics(frame_records, cam_id, mode)

    # Per-video metrics (scalars)
    metrics: Dict[str, Any] = {}
    metrics.update(detection_frame_stats(frame_records))
    metrics.update(detection_temporal_persistence(frame_records))
    metrics.update(detection_confidence_stats(frame_records))

    if tracks:
        metrics.update(pose_confidence_stats(tracks))
        metrics.update(pose_limb_length_consistency(tracks))
        metrics.update(pose_temporal_smoothness(tracks))
        metrics.update(pose_temporal_cosine(tracks))
        metrics.update(tracking_basic_stats(tracks, total_frames=total_frames))
        metrics.update(reid_temporal_embedding_std(tracks))
        metrics.update(reid_embedding_dispersion(tracks))

    metrics.update(pipeline_stage_yield(frame_records))

    metrics.update({
        "mode": mode,
        "cam_id": cam_id,
        "n_frames": int(total_frames),
        "proxy_id_switches": int(total_proxy_sw),
        "proxy_id_checks": int(total_proxy_chk),
    })

    return per_track_df, limb_df, jitter_df, per_frame_df, metrics


# ---------------------------------------------------------------------------
# Driver for RTMO root
# ---------------------------------------------------------------------------

def _infer_run_info(out_root: Path, cam_files) -> None:
    """Set NUM_JOINTS / MODE from run_config.yaml, else from the first record."""
    global NUM_JOINTS, MODE
    cfg_path = out_root / "run_config.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        MODE = str(cfg.get("pose", {}).get("type", MODE))
    for cf in cam_files:
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                kps = json.loads(line).get("keypoints") or []
                if kps:
                    NUM_JOINTS = len(kps) if isinstance(kps[0], (list, tuple)) else len(kps) // 3
                    return


def run_eval(root: str, global_file: str = DEFAULT_GLOBAL_FILE):
    out_root = Path(root)
    cam_files = sorted(out_root.rglob("*_poses_union.jsonl"))
    if not cam_files:
        print(f"[ERROR] No *_poses_union.jsonl under {out_root}")
        return
    _infer_run_info(out_root, cam_files)
    print(f"[INFO] pose={MODE} joints={NUM_JOINTS}")

    print(f"[INFO] Found {len(cam_files)} pose files:")
    for cf in cam_files:
        print("   -", cf.relative_to(out_root))

    all_track_dfs = []
    all_limb_dfs = []
    all_jitter_dfs = []
    all_frame_dfs = []
    all_video_rows = []

    for cf in cam_files:
        cam_name = cf.name.split("_")[0]
        print(f"[INFO] Evaluating {cf}")
        frame_records = build_frame_records_rtmo(cf)
        per_track_df, limb_df, jitter_df, per_frame_df, metrics = compute_all_metrics_for_file(
            frame_records, cam_id=cam_name, mode=MODE
        )
        rel_file = str(cf.relative_to(out_root))
        metrics["file"] = rel_file

        if not per_track_df.empty:
            per_track_df["file"] = rel_file
            all_track_dfs.append(per_track_df)

        if not limb_df.empty:
            limb_df["file"] = rel_file
            limb_df["mode"] = MODE
            limb_df["cam_id"] = cam_name
            all_limb_dfs.append(limb_df)

        if not jitter_df.empty:
            jitter_df["file"] = rel_file
            jitter_df["mode"] = MODE
            jitter_df["cam_id"] = cam_name
            all_jitter_dfs.append(jitter_df)

        if not per_frame_df.empty:
            per_frame_df["file"] = rel_file
            all_frame_dfs.append(per_frame_df)

        all_video_rows.append(metrics)

    outdir = out_root
    outdir.mkdir(parents=True, exist_ok=True)

    per_track = pd.concat(all_track_dfs, ignore_index=True) if all_track_dfs else pd.DataFrame()
    per_video = pd.DataFrame(all_video_rows)
    limb_all = pd.concat(all_limb_dfs, ignore_index=True) if all_limb_dfs else pd.DataFrame()
    jitter_all = pd.concat(all_jitter_dfs, ignore_index=True) if all_jitter_dfs else pd.DataFrame()
    frame_all = pd.concat(all_frame_dfs, ignore_index=True) if all_frame_dfs else pd.DataFrame()

    # overall: mean of numeric metrics, sum for obvious counts
    if not per_video.empty:
        overall: Dict[str, Any] = {}
        num_cols = per_video.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in {"n_frames", "total_detections", "num_tracks",
                       "proxy_id_switches", "proxy_id_checks"}:
                overall[col] = float(per_video[col].sum(skipna=True))
            else:
                overall[col] = float(per_video[col].mean(skipna=True))
        overall["mode"] = MODE
    else:
        overall = {"mode": MODE}

    per_track.to_csv(outdir / "per_track.csv", index=False)
    per_video.to_csv(outdir / "per_video.csv", index=False)
    pd.DataFrame([overall]).to_csv(outdir / "overall.csv", index=False)
    limb_all.to_csv(outdir / "limb_lengths.csv", index=False)
    jitter_all.to_csv(outdir / "jitter_timeseries.csv", index=False)
    frame_all.to_csv(outdir / "per_frame_metrics.csv", index=False)

    print(f"[OK] Wrote {outdir/'per_track.csv'}")
    print(f"[OK] Wrote {outdir/'per_video.csv'}")
    print(f"[OK] Wrote {outdir/'overall.csv'}")
    print(f"[OK] Wrote {outdir/'limb_lengths.csv'}")
    print(f"[OK] Wrote {outdir/'jitter_timeseries.csv'}")
    print(f"[OK] Wrote {outdir/'per_frame_metrics.csv'}")

    # cross-camera scalar metrics
    global_path = out_root / global_file
    crosscam_metrics = compute_crosscam_metrics(global_path)
    if crosscam_metrics:
        pd.DataFrame([crosscam_metrics]).to_csv(outdir / "crosscam.csv", index=False)
        print(f"[OK] Wrote {outdir/'crosscam.csv'}")
    else:
        print(f"[WARN] No cross-camera metrics (global file missing or empty)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised metrics for one posepipe run directory."
    )
    parser.add_argument(
        "--root", "--rtmo-root",
        dest="rtmo_root",
        type=str,
        required=True,
        help="Run directory written by `posepipe run`.",
    )
    parser.add_argument(
        "--global-file",
        type=str,
        default=DEFAULT_GLOBAL_FILE,
        help="Global tracks JSONL filename (under RTMO root).",
    )
    args = parser.parse_args()
    run_eval(args.rtmo_root, args.global_file)


run_rtmo = run_eval  # old name


if __name__ == "__main__":
    main()
