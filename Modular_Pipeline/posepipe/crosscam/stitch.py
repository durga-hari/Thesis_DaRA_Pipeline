"""
Cross-camera hybrid union-find stitching (ported from RTMO_DeepSORT_Pipeline).

1. Per camera: read <cam>/results.jsonl, drop very short tracks, average and
   L2-normalise each track's embedding, average its pose.
2. Intra-camera merge: chain fragments whose mean embeddings have cosine
   distance <= intra_emb_thr, whose time gap is <= intra_time_gap frames and
   (optionally) whose boundary boxes overlap with IoU >= intra_iou_thr.
3. Cross-camera: hybrid similarity alpha*ReID + (1-alpha)*pose, mutual best
   match above cos_thr, union-find for transitivity.
4. Write crosscam_map.json  {cam_dir: {local_track_id: global_id}}.

Fixes relative to the original script:
- The map is keyed by the *original* local track IDs. The old version keyed
  it by the index of the merged group ("0", "1", ...), so inject_global_ids
  attached global IDs to the wrong local tracks.
- The intra-merge thresholds come from the config / param study; they were
  hard-coded (0.85 similarity, 30 frames) and the IoU gate did not exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def _l2(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ub = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (ua + ub - inter + 1e-12))


def read_pose_jsonl(path: Path, min_frames: int = 3):
    """{track_id: [(frame, emb, kps, bbox), ...]} for tracks with >= min_frames."""
    data = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            emb = np.asarray(item.get("feature") or item.get("embedding") or [], dtype=np.float32)
            kps = np.asarray(item.get("keypoints") or [], dtype=np.float32)
            if emb.size == 0 or np.any(np.isnan(emb)) or kps.size == 0:
                continue
            data[item["id"]].append((int(item["frame"]), _l2(emb), kps, item.get("bbox")))
    kept = {tid: v for tid, v in data.items() if len(v) >= min_frames}
    if len(kept) < len(data):
        print(f"[Stitch] {path.parent.name}: dropped {len(data) - len(kept)} tracks < {min_frames} frames")
    return kept


def build_track_embeds(data):
    tracks = {}
    for tid, rows in data.items():
        rows = sorted(rows, key=lambda r: r[0])
        tracks[tid] = {
            "start": rows[0][0], "end": rows[-1][0],
            "start_box": rows[0][3], "end_box": rows[-1][3],
            "emb": _l2(np.mean(np.stack([r[1] for r in rows]), axis=0)),
            "pose": np.mean(np.stack([r[2] for r in rows]), axis=0),
        }
    return tracks


def merge_local_tracks(tracks, emb_thr: float = 0.15, time_gap: int = 30, iou_thr: float = 0.0):
    """Greedy intra-camera merge. Returns {group_key: {..., "members": [tids]}}."""
    tids = sorted(tracks, key=lambda t: tracks[t]["start"])
    used, merged = set(), {}
    for i, t1 in enumerate(tids):
        if t1 in used:
            continue
        group = [t1]
        for t2 in tids[i + 1:]:
            if t2 in used:
                continue
            a, b = tracks[t1], tracks[t2]
            sim = float(np.dot(a["emb"], b["emb"]))
            if sim < 1.0 - emb_thr or abs(b["start"] - a["end"]) > time_gap:
                continue
            if iou_thr > 0 and (a["end_box"] is None or b["start_box"] is None
                                or _iou(a["end_box"], b["start_box"]) < iou_thr):
                continue
            group.append(t2)
            used.add(t2)
        merged[str(t1)] = {
            "start": min(tracks[t]["start"] for t in group),
            "end": max(tracks[t]["end"] for t in group),
            "emb": _l2(np.mean(np.stack([tracks[t]["emb"] for t in group]), axis=0)),
            "pose": np.mean([tracks[t]["pose"] for t in group], axis=0),
            "members": group,
        }
    return merged


def hybrid_similarity(vi, vj, pi, pj, alpha=0.6):
    sim_reid = float(np.dot(vi, vj))
    if pi.shape == pj.shape:
        sim_pose = float(np.exp(-np.linalg.norm(pi - pj) / 100.0))
    else:
        sim_pose = 0.0
    return alpha * sim_reid + (1 - alpha) * sim_pose


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def stitch(root, cos_thr: float = 0.4, alpha: float = 0.5, intra_emb_thr: float = 0.15,
           intra_time_gap: int = 30, intra_iou_thr: float = 0.0, min_frames: int = 3) -> dict:
    root = Path(root)
    files = sorted(root.glob("*/results.jsonl"))
    if not files:
        print(f"[Stitch] no */results.jsonl under {root}")
        return {}

    cams = {}
    for f in files:
        tracks = build_track_embeds(read_pose_jsonl(f, min_frames=min_frames))
        cams[f.parent.name] = merge_local_tracks(tracks, intra_emb_thr, intra_time_gap, intra_iou_thr)
        print(f"[Stitch] {f.parent.name}: {len(tracks)} local tracks -> {len(cams[f.parent.name])} after intra-merge")

    uf = UnionFind()
    key = lambda cam, g: f"{cam}#{g}"
    for cam in cams:
        for g in cams[cam]:
            uf.find(key(cam, g))

    for camA, camB in combinations(cams, 2):
        ta, tb = cams[camA], cams[camB]
        ka, kb = list(ta), list(tb)
        if not ka or not kb:
            continue
        mat = np.array([[hybrid_similarity(ta[a]["emb"], tb[b]["emb"], ta[a]["pose"], tb[b]["pose"], alpha)
                         for b in kb] for a in ka], dtype=np.float32)
        best_j, best_i = mat.argmax(1), mat.argmax(0)
        for i, j in enumerate(best_j):
            if best_i[j] == i and mat[i, j] >= cos_thr:
                uf.union(key(camA, ka[i]), key(camB, kb[j]))

    comps = defaultdict(list)
    for cam in cams:
        for g in cams[cam]:
            comps[uf.find(key(cam, g))].append((cam, g))

    global_map = defaultdict(dict)
    rows = []
    for gid, members in enumerate(comps.values(), 1):
        locals_ = []
        for cam, g in members:
            for tid in cams[cam][g]["members"]:
                global_map[cam][str(tid)] = gid
                locals_.append(f"{cam}:{tid}")
        rows.append([gid, "; ".join(locals_)])

    with open(root / "crosscam_map.json", "w", encoding="utf-8") as f:
        json.dump(global_map, f, indent=2)
    with open(root / "crosscam_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "members"])
        w.writerows(rows)
    print(f"[Stitch] {len(comps)} global identities -> {root / 'crosscam_map.json'}")
    return dict(global_map)


def main(args):
    """Namespace entry point (kept compatible with the old param-study call)."""
    return stitch(args.input_root, cos_thr=args.cos_thr, alpha=args.alpha,
                  intra_emb_thr=getattr(args, "intra_merge_emb_thr", 0.15),
                  intra_time_gap=getattr(args, "intra_merge_time_gap", 30),
                  intra_iou_thr=getattr(args, "intra_merge_iou_thr", 0.0),
                  min_frames=getattr(args, "min_frames", 3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid union-find cross-camera stitching")
    p.add_argument("--input-root", "--root", dest="input_root", required=True)
    p.add_argument("--cos-thr", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--intra-merge-emb-thr", type=float, default=0.15)
    p.add_argument("--intra-merge-time-gap", type=int, default=30)
    p.add_argument("--intra-merge-iou-thr", type=float, default=0.0)
    main(p.parse_args())
