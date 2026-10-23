"""
Apply the cross-camera map: per-camera *_poses_union.jsonl with global_id,
one merged_global_tracks_union.jsonl, and local/global-ID annotated videos.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from ..io import read_jsonl
from ..visualize import color_for_id, draw_pose_box_and_id

CAM_SUFFIXES = ("_output", "_poses", "_cam")


def cam_key(dir_name: str) -> str:
    for suf in CAM_SUFFIXES:
        if dir_name.endswith(suf):
            return dir_name[: -len(suf)]
    return dir_name


def inject_global_ids(out_root: Path, crosscam_map: dict) -> None:
    for cam_dir in sorted(Path(out_root).glob("*")):
        src = cam_dir / "results.jsonl"
        if not src.exists():
            continue
        local2global = crosscam_map.get(cam_dir.name, crosscam_map.get(cam_key(cam_dir.name), {}))
        dst = cam_dir / f"{cam_dir.name}_poses_union.jsonl"
        mapped = total = 0
        with open(dst, "w", encoding="utf-8") as fw:
            for item in read_jsonl(src):
                total += 1
                gid = local2global.get(str(item.get("id")))
                if gid is not None:
                    item["global_id"] = gid
                    mapped += 1
                fw.write(json.dumps(item) + "\n")
        print(f"[Union] {dst.name}: {mapped}/{total} records have a global_id")


def merge_tracks_union(out_root: Path, crosscam_map: dict, out_path: Path,
                       min_frames_per_gid: int = 80, require_multi_cam: bool = False) -> None:
    records, gid_cams, gid_frames = [], defaultdict(set), defaultdict(int)
    for jf in sorted(Path(out_root).glob("*/*_poses_union.jsonl")):
        cam = jf.parent.name
        for item in read_jsonl(jf):
            gid = item.get("global_id")
            if gid is None:
                continue
            records.append({
                "cam_id": cam, "frame": int(item.get("frame", 0)), "local_id": item.get("id"),
                "global_id": gid, "bbox": item.get("bbox"), "keypoints": item.get("keypoints"),
                "score": item.get("score"), "pose_conf": item.get("pose_conf"),
            })
            gid_cams[str(gid)].add(cam)
            gid_frames[str(gid)] += 1

    keep = {g for g, n in gid_frames.items()
            if n >= min_frames_per_gid and (not require_multi_cam or len(gid_cams[g]) >= 2)}
    kept = 0
    with open(out_path, "w", encoding="utf-8") as fw:
        for r in records:
            if str(r["global_id"]) in keep:
                fw.write(json.dumps(r) + "\n")
                kept += 1
    print(f"[Merge] {out_path.name}: {kept} records, {len(keep)} global IDs after filters")


def _load_by_frame(path: Path):
    by_frame = defaultdict(list)
    for item in read_jsonl(path):
        by_frame[int(item.get("frame", 0))].append(item)
    return by_frame


def annotate_videos_with_ids(out_root: Path, video_paths: Dict[str, Path],
                             start_frame: int = 0) -> None:
    """video_paths: {video_stem: path}. Writes <cam>_localids.mp4 / <cam>_globalids.mp4."""
    for cam_dir in sorted(Path(out_root).glob("*")):
        if not cam_dir.is_dir():
            continue
        video: Optional[Path] = video_paths.get(cam_key(cam_dir.name))
        json_local = cam_dir / "results.jsonl"
        json_global = cam_dir / f"{cam_dir.name}_poses_union.jsonl"
        if not (video and Path(video).exists() and json_local.exists() and json_global.exists()):
            print(f"[Annotate] skipping {cam_dir.name} (missing video or JSONL)")
            continue

        for mode, path in (("local", json_local), ("global", json_global)):
            data = _load_by_frame(path)
            cap = cv2.VideoCapture(str(video))
            if start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            last = max(data) if data else start_frame
            out_vid = cam_dir / f"{cam_dir.name}_{mode}ids.mp4"
            writer = cv2.VideoWriter(str(out_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for fid in tqdm(range(start_frame + 1, last + 1), desc=f"{cam_dir.name} ({mode})"):
                ok, frame = cap.read()
                if not ok:
                    break
                for item in data.get(fid, []):
                    idv = item.get("id") if mode == "local" else item.get("global_id")
                    if idv is None:
                        continue
                    tag = "LID" if mode == "local" else "GID"
                    draw_pose_box_and_id(frame, np.asarray(item.get("keypoints") or [], np.float32),
                                         int(idv), color=color_for_id(int(idv), mode),
                                         bbox=item.get("bbox"), label=f"{tag} {int(idv)}")
                writer.write(frame)
            cap.release()
            writer.release()
            print(f"[Annotate] {out_vid}")
