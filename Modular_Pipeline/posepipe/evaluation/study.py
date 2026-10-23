#!/usr/bin/env python3
"""
Parameter study runner.

Pose-backend aware: the pose model comes from the config (or --pose), and
outputs go to <out_root>/study/<pose>/<variant>/<experiment>/.

- every swept parameter (tracker, intra-camera merge, stitching, global filters) is applied
- windows without --make-clips seek to the window start
- per-run videos are off unless --keep-videos
- variants are any <tracker>_<reid> pair (deepsort_resnet, strongsort_osnet,
  deepsort_osnet, strongsort_resnet) or both/all

New capability:
- Run multiple experiments in one invocation:
    --experiments all
    --experiments stitch_cos_thr,stitch_alpha,tracker_ds_max_age

Output layout (unchanged, stable):
  <out_root>/study/<pose>/<variant>/<experiment>/
    study_summary.csv
    <variant>/window_s*_n*/
      runs/<tag>/
      plots/
      clips_s*_n*/ (if --make-clips)

Important behavior:
- Variants are still separated (deepsort_resnet vs strongsort_osnet)
- Experiments are executed sequentially (no parallelism)
- Plots are updated incrementally after each run (AP-like)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import matplotlib.pyplot as plt

from .. import runner as _runner
from ..config import component_cfg, load_config
from ..crosscam import merge as _merge
from ..crosscam.stitch import main as _stitch_main


class R:  # adapter so the original call sites stay readable
    process_video = staticmethod(_runner.process_video)
    crosscam_union_stitch_main = staticmethod(_stitch_main)
    inject_global_ids = staticmethod(_merge.inject_global_ids)
    merge_tracks_union = staticmethod(_merge.merge_tracks_union)


VARIANTS = ["deepsort_resnet", "strongsort_osnet", "deepsort_osnet", "strongsort_resnet"]
_REID_TYPE = {"resnet": "resnet50", "osnet": "osnet"}


def _nest_overrides(overrides: dict, cfg: dict) -> dict:
    """{"tracker": {"type": "deepsort", "max_age": 60}} ->
       {"tracker": {"type": "deepsort", "deepsort": {"max_age": 60}}}"""
    out = deepcopy(overrides)
    for section, ov in out.items():
        if section not in ("detector", "pose", "reid", "tracker") or not isinstance(ov, dict):
            continue
        t = ov.get("type", cfg.get(section, {}).get("type"))
        flat = {k: v for k, v in ov.items() if k != "type" and not isinstance(v, dict)}
        for k in flat:
            ov.pop(k)
        if flat:
            ov.setdefault(t, {}).update(flat)
    return out


# --------------------------
# YAML + dict helpers
# --------------------------

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


# --------------------------
# Windowing
# --------------------------

def parse_windows(s: str) -> List[Tuple[int, int]]:
    """
    Parse "start:end,start:end,..." -> [(start, length), ...]
    """
    out: List[Tuple[int, int]] = []
    s = (s or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        a, b = p.split(":")
        start = int(a)
        end = int(b)
        if end <= start:
            raise ValueError(f"Bad window '{p}' (end must be > start).")
        out.append((start, end - start))
    return out

def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    logging.debug("CMD: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ffmpeg_exists() -> bool:
    try:
        run_cmd(["ffmpeg", "-version"], check=True)
        return True
    except Exception:
        return False

def list_session_videos(video_path: Path) -> List[Path]:
    if video_path.is_dir():
        return sorted(video_path.glob("*.mp4"))
    return [video_path]

def make_clip_ffmpeg(
    src_video: Path,
    dst_video: Path,
    start_frame: int,
    length: int,
    fps_hint: Optional[float] = None,
) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found. Install ffmpeg or run without --make-clips.")

    fps = fps_hint
    if fps is None:
        try:
            pr = run_cmd([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(src_video)
            ], check=True)
            rate = pr.stdout.strip()
            if "/" in rate:
                num, den = rate.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(rate)
        except Exception:
            fps = 25.0

    start_sec = start_frame / float(fps)
    dur_sec = length / float(fps)

    dst_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_video),
        "-ss", f"{start_sec:.6f}",
        "-t", f"{dur_sec:.6f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(dst_video),
    ]
    pr = run_cmd(cmd, check=True)
    if pr.stderr:
        logging.debug("ffmpeg stderr (trim): %s", pr.stderr[-1200:])

def prepare_window_videos(
    session_dir: Path,
    window: Tuple[int, int],
    work_dir: Path,
    make_clips: bool,
) -> Path:
    start, length = window
    if not make_clips:
        return session_dir

    clips_dir = work_dir / f"clips_s{start}_n{length}"
    if clips_dir.exists():
        return clips_dir

    clips_dir.mkdir(parents=True, exist_ok=True)
    vids = list_session_videos(session_dir)
    if not vids:
        raise RuntimeError(f"No videos found in {session_dir}")

    logging.info("Creating window clips: start=%d len=%d -> %s", start, length, clips_dir)
    for v in vids:
        dst = clips_dir / v.name
        make_clip_ffmpeg(v, dst, start_frame=start, length=length)
    return clips_dir


# --------------------------
# Metrics helpers
# --------------------------

def safe_json_load(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}

def count_global_ids_in_merged(merged_jsonl: Path) -> int:
    if not merged_jsonl.exists():
        return 0
    seen = set()
    with merged_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            gid = rec.get("global_id", None)
            if gid is None:
                continue
            try:
                seen.add(int(gid))
            except Exception:
                pass
    return len(seen)

def component_sizes_from_union_map(union_map: Dict[str, Dict[str, int]]) -> Dict[int, int]:
    sizes: Dict[int, int] = {}
    for _, m in (union_map or {}).items():
        if not isinstance(m, dict):
            continue
        for _, gid in m.items():
            try:
                gid_int = int(gid)
            except Exception:
                continue
            sizes[gid_int] = sizes.get(gid_int, 0) + 1
    return sizes

def summarize_local_tracks_total(per_cam_local_ids_json: str) -> int:
    d = safe_json_load(per_cam_local_ids_json)
    if not isinstance(d, dict):
        return 0
    s = 0
    for _, v in d.items():
        try:
            s += int(v)
        except Exception:
            pass
    return s


# --------------------------
# Plotting (AP-style)
# --------------------------

def get_sweep_x_and_label(experiment: str, row: Dict[str, Any]) -> Tuple[Optional[float], str]:
    exp = experiment.lower().strip()
    tracker_params = safe_json_load(row.get("tracker_params_json", ""))

    if exp == "stitch_cos_thr":
        return float(row["stitch_cos_thr"]), "stitch_cos_thr"
    if exp == "stitch_alpha":
        return float(row["stitch_alpha"]), "stitch_alpha"

    if exp == "intra_merge_emb_thr":
        return float(row["intra_merge_emb_thr"]), "intra_merge_emb_thr"
    if exp == "intra_merge_iou_thr":
        return float(row["intra_merge_iou_thr"]), "intra_merge_iou_thr"
    if exp == "intra_merge_time_gap":
        return float(row["intra_merge_time_gap"]), "intra_merge_time_gap"

    if exp == "global_min_frames":
        return float(row["min_frames_per_gid"]), "min_frames_per_gid"
    if exp == "global_require_multi_cam":
        return float(row["require_multi_cam"]), "require_multi_cam"

    if exp in ("tracker_ds_max_age", "tracker_ss_ttl"):
        v = tracker_params.get("max_age", None)
        return (float(v) if v is not None else None), "max_age"
    if exp in ("tracker_ds_max_cosine", "tracker_ss_sim_thr"):
        v = tracker_params.get("max_cosine_distance", None)
        return (float(v) if v is not None else None), "max_cosine_distance"
    if exp in ("tracker_ds_max_iou", "tracker_ss_iou_thr"):
        v = tracker_params.get("max_iou_distance", None)
        return (float(v) if v is not None else None), "max_iou_distance"
    if exp == "tracker_ds_n_init":
        v = tracker_params.get("n_init", None)
        return (float(v) if v is not None else None), "n_init"

    return None, "parameter"

def save_plots_for_window_variant(
    rows: List[Dict[str, Any]],
    experiment: str,
    out_dir: Path,
    title_prefix: str,
) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    xs: List[float] = []
    gids: List[int] = []
    lcs: List[int] = []
    ltot: List[int] = []
    x_label = "parameter"

    for r in rows:
        x, lbl = get_sweep_x_and_label(experiment, r)
        if x is None:
            continue
        x_label = lbl
        xs.append(float(x))
        gids.append(int(float(r["global_ids"])))
        lcs.append(int(float(r["largest_component_local_tracks"])))
        ltot.append(summarize_local_tracks_total(r.get("per_cam_local_ids", "")))

    if not xs:
        return

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    gids = [gids[i] for i in order]
    lcs = [lcs[i] for i in order]
    ltot = [ltot[i] for i in order]

    plt.figure(figsize=(7, 4))
    plt.plot(xs, gids, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("global_ids")
    plt.title(f"{title_prefix} | global_ids")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_ids_vs_param.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(xs, lcs, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("largest_component_local_tracks")
    plt.title(f"{title_prefix} | largest_component")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "largest_component_vs_param.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ltot, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("sum_local_tracks_all_cams")
    plt.title(f"{title_prefix} | total_local_tracks")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "total_local_tracks_vs_param.png", dpi=150)
    plt.close()


# --------------------------
# Variant handling
# --------------------------

def apply_variant(cfg: dict, variant: str, osnet_weights: Optional[str], resnet_weights: Optional[str],
                  keep_videos: bool = False) -> dict:
    v = variant.lower().strip()
    if v not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Valid: {', '.join(VARIANTS)}")
    trk, reid_key = v.split("_")
    cfg = deepcopy(cfg)
    cfg.setdefault("reid", {})
    cfg.setdefault("tracker", {})
    cfg.setdefault("pipeline", {})
    cfg["tracker"]["type"] = trk
    cfg["reid"]["type"] = _REID_TYPE[reid_key]
    w = osnet_weights if reid_key == "osnet" else resnet_weights
    if w:
        cfg["reid"].setdefault(_REID_TYPE[reid_key], {})["weights"] = w

    cfg["pipeline"]["save_json"] = True
    cfg["pipeline"]["annotate_ids"] = False
    if not keep_videos:
        cfg["pipeline"]["save_video"] = False
        cfg["pipeline"]["draw"] = False
    return cfg


# --------------------------
# Experiment specs
# --------------------------

ALL_EXPERIMENTS: List[str] = [
    "stitch_cos_thr",
    "stitch_alpha",
    "intra_merge_time_gap",
    "intra_merge_emb_thr",
    "intra_merge_iou_thr",
    "tracker_ds_max_age",
    "tracker_ds_max_cosine",
    "tracker_ds_max_iou",
    "tracker_ds_n_init",
    "tracker_ss_sim_thr",
    "tracker_ss_iou_thr",
    "tracker_ss_ttl",
    "global_min_frames",
    "global_require_multi_cam",
]

@dataclass(frozen=True)
class RunSpec:
    tag: str
    cfg_overrides: Dict[str, Any]
    intra_merge_emb_thr: float
    intra_merge_iou_thr: float
    intra_merge_time_gap: int
    stitch_cos_thr: float
    stitch_alpha: float
    min_frames_per_gid: int
    require_multi_cam: bool

def get_experiment_runs(experiment: str) -> List[RunSpec]:
    exp = experiment.lower().strip()

    def_intra_e = 0.15
    def_intra_i = 0.0   # matches configs/default.yaml crosscam.intra_merge
    def_intra_g = 30    # matches configs/default.yaml
    def_cos = 0.40
    def_alpha = 0.50
    def_min_gid = 80
    def_req_mc = False

    if exp == "tracker_ds_max_age":
        ages = [30, 60, 90, 120, 180]
        return [RunSpec(f"ds_max_age_{a}", {"tracker": {"type": "deepsort", "max_age": a}},
                        def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for a in ages]

    if exp == "tracker_ds_max_cosine":
        vals = [0.15, 0.20, 0.25, 0.30, 0.35]
        return [RunSpec(f"ds_max_cos_{v:.2f}", {"tracker": {"type": "deepsort", "max_cosine_distance": v}},
                        def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "tracker_ds_max_iou":
        vals = [0.5, 0.7, 0.9]
        return [RunSpec(f"ds_max_iou_{v:.2f}", {"tracker": {"type": "deepsort", "max_iou_distance": v}},
                        def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "tracker_ds_n_init":
        vals = [1, 3, 5]
        return [RunSpec(f"ds_n_init_{v}", {"tracker": {"type": "deepsort", "n_init": v}},
                        def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "tracker_ss_sim_thr":
        vals = [0.35, 0.40, 0.45, 0.50, 0.55]
        return [RunSpec(f"ss_sim_{v:.2f}", {"tracker": {"type": "strongsort", "max_cosine_distance": v}},
                        def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "tracker_ss_iou_thr":
        vals = [0.20, 0.30, 0.40, 0.50]
        return [RunSpec(f"ss_iou_{v:.2f}", {"tracker": {"type": "strongsort", "max_iou_distance": v}},
                        def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "tracker_ss_ttl":
        vals = [40, 60, 90, 120, 180]
        return [RunSpec(f"ss_ttl_{v}", {"tracker": {"type": "strongsort", "max_age": v}},
                        def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "stitch_cos_thr":
        thr = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
        return [RunSpec(f"st_cos_{t:.2f}", {}, def_intra_e, def_intra_i, def_intra_g, t, def_alpha, def_min_gid, def_req_mc) for t in thr]

    if exp == "stitch_alpha":
        alphas = [1.0, 0.8, 0.6, 0.5, 0.4]
        return [RunSpec(f"st_alpha_{a:.2f}", {}, def_intra_e, def_intra_i, def_intra_g, def_cos, a, def_min_gid, def_req_mc) for a in alphas]

    if exp == "intra_merge_emb_thr":
        vals = [0.10, 0.15, 0.20, 0.25, 0.30]
        return [RunSpec(f"intraE_{v:.2f}", {}, v, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "intra_merge_iou_thr":
        vals = [0.20, 0.25, 0.30, 0.35]
        return [RunSpec(f"intraI_{v:.2f}", {}, def_intra_e, v, def_intra_g, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "intra_merge_time_gap":
        vals = [30, 75, 120, 180]
        return [RunSpec(f"intraG_{v}", {}, def_intra_e, def_intra_i, v, def_cos, def_alpha, def_min_gid, def_req_mc) for v in vals]

    if exp == "global_min_frames":
        vals = [1, 30, 80, 150, 300]
        return [RunSpec(f"gid_min_{v}", {}, def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, v, def_req_mc) for v in vals]

    if exp == "global_require_multi_cam":
        vals = [False, True]
        return [RunSpec(f"gid_reqMC_{int(v)}", {}, def_intra_e, def_intra_i, def_intra_g, def_cos, def_alpha, def_min_gid, v) for v in vals]

    raise ValueError(f"Unknown experiment '{experiment}'. Valid: {', '.join(ALL_EXPERIMENTS)}")

def is_run_compatible_with_variant(run: RunSpec, variant: str) -> bool:
    trk = (run.cfg_overrides.get("tracker", {}).get("type") or "").lower()
    if not trk:
        return True
    return trk == variant.lower().strip().split("_")[0]


# --------------------------
# One run
# --------------------------

def run_one(base_cfg: dict, run: RunSpec, window_dir: Path, out_root: Path, max_frames_override: Optional[int]) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    deep_update(cfg, _nest_overrides(run.cfg_overrides, cfg))

    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["intra_merge_emb_thr"] = float(run.intra_merge_emb_thr)
    cfg["pipeline"]["intra_merge_iou_thr"] = float(run.intra_merge_iou_thr)
    cfg["pipeline"]["intra_merge_time_gap"] = int(run.intra_merge_time_gap)

    if max_frames_override is not None:
        cfg["pipeline"]["max_frames"] = int(max_frames_override)

    cfg["out_root"] = str(out_root)

    dump_yaml(cfg, out_root / "run_config.yaml")

    vids = list_session_videos(window_dir)
    if not vids:
        raise RuntimeError(f"No videos to process in {window_dir}")

    per_cam_local_ids: Dict[str, int] = {}

    for v in vids:
        cam_out = out_root / f"{v.stem}_output"
        cam_out.mkdir(parents=True, exist_ok=True)

        logging.info("[Run %s] Processing %s", run.tag, v.name)
        R.process_video(v, cam_out, cfg)

        results_jsonl = cam_out / "results.jsonl"
        if results_jsonl.exists():
            seen = set()
            with results_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    tid = rec.get("id", None)
                    if tid is None:
                        continue
                    try:
                        seen.add(int(tid))
                    except Exception:
                        pass
            per_cam_local_ids[cam_out.name] = len(seen)
        else:
            per_cam_local_ids[cam_out.name] = 0

    import types
    args_union = types.SimpleNamespace(
        input_root=str(out_root),
        cos_thr=float(run.stitch_cos_thr),
        alpha=float(run.stitch_alpha),
        intra_merge_emb_thr=float(run.intra_merge_emb_thr),
        intra_merge_iou_thr=float(run.intra_merge_iou_thr),
        intra_merge_time_gap=int(run.intra_merge_time_gap),
        min_frames=int(cfg.get("crosscam", {}).get("min_frames", 3)),
    )
    R.crosscam_union_stitch_main(args_union)

    map_file = out_root / "crosscam_map.json"
    union_map: Dict[str, Dict[str, int]] = {}
    if map_file.exists():
        with map_file.open("r", encoding="utf-8") as f:
            union_map = json.load(f)
    else:
        logging.warning("Missing crosscam_map.json for run %s (no globals).", run.tag)

    R.inject_global_ids(out_root, union_map)

    merged_path = out_root / "merged_global_tracks_union.jsonl"
    if union_map:
        R.merge_tracks_union(
            out_root,
            union_map,
            merged_path,
            min_frames_per_gid=int(run.min_frames_per_gid),
            require_multi_cam=bool(run.require_multi_cam),
        )
    else:
        merged_path.write_text("", encoding="utf-8")

    gid_count = count_global_ids_in_merged(merged_path)
    comp_sizes = component_sizes_from_union_map(union_map)
    largest_comp = max(comp_sizes.values()) if comp_sizes else 0

    tracker_params = component_cfg(cfg["tracker"])
    tracker_params_json = json.dumps(tracker_params, sort_keys=True)
    reid_params = component_cfg(cfg["reid"])

    return {
        "tag": run.tag,
        "pose": str(cfg["pose"]["type"]).lower(),
        "tracker": str(tracker_params["type"]).lower(),
        "reid": str(reid_params["type"]).lower(),
        "reid_weights": str(reid_params.get("weights", "")),
        "tracker_params_json": tracker_params_json,

        "intra_merge_emb_thr": run.intra_merge_emb_thr,
        "intra_merge_iou_thr": run.intra_merge_iou_thr,
        "intra_merge_time_gap": run.intra_merge_time_gap,

        "stitch_cos_thr": run.stitch_cos_thr,
        "stitch_alpha": run.stitch_alpha,

        "min_frames_per_gid": run.min_frames_per_gid,
        "require_multi_cam": int(run.require_multi_cam),

        "global_ids": gid_count,
        "largest_component_local_tracks": largest_comp,
        "per_cam_local_ids": json.dumps(per_cam_local_ids, sort_keys=True),

        "out_dir": str(out_root),
    }


# --------------------------
# Run one experiment end-to-end
# --------------------------

def run_experiment(
    base_cfg_raw: dict,
    session_dir: Path,
    base_out: Path,
    experiment: str,
    variant: str,
    windows: List[Tuple[int, int]],
    make_clips: bool,
    clean: bool,
    study_out_override: Optional[str],
    osnet_w: Optional[str],
    resnet_w: Optional[str],
    max_frames_cli: int,
    keep_videos: bool = False,
) -> None:
    # Per-experiment output root (stable)
    if study_out_override:
        study_out = Path(study_out_override).resolve()
    else:
        study_out = base_out / "study" / str(base_cfg_raw["pose"]["type"]).lower() / variant / experiment

    if clean and study_out.exists():
        shutil.rmtree(study_out)
    study_out.mkdir(parents=True, exist_ok=True)

    runs = get_experiment_runs(experiment)
    base_cfg_variant = apply_variant(base_cfg_raw, variant=variant, osnet_weights=osnet_w,
                                     resnet_weights=resnet_w, keep_videos=keep_videos)

    summary_path = study_out / "study_summary.csv"
    all_rows: List[Dict[str, Any]] = []

    with summary_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "variant",
                "window_start", "window_len",
                "tag",
                "pose", "tracker", "reid", "reid_weights",
                "tracker_params_json",
                "intra_merge_emb_thr", "intra_merge_iou_thr", "intra_merge_time_gap",
                "stitch_cos_thr", "stitch_alpha",
                "min_frames_per_gid", "require_multi_cam",
                "global_ids",
                "largest_component_local_tracks",
                "per_cam_local_ids",
                "out_dir",
            ],
        )
        writer.writeheader()

        for (w_start, w_len) in windows:
            window_work = study_out / variant / f"window_s{w_start}_n{w_len}"
            window_work.mkdir(parents=True, exist_ok=True)

            window_dir = prepare_window_videos(
                session_dir=session_dir,
                window=(w_start, w_len),
                work_dir=window_work,
                make_clips=make_clips,
            )

            max_frames_override = max_frames_cli if max_frames_cli > 0 else w_len

            for run in runs:
                if not is_run_compatible_with_variant(run, variant):
                    continue

                run_out = window_work / "runs" / run.tag
                if run_out.exists():
                    logging.info("Skipping existing run output: %s", run_out)
                    continue
                run_out.mkdir(parents=True, exist_ok=True)

                cfg_run = deepcopy(base_cfg_variant)
                cfg_run["video_dir"] = str(window_dir)
                cfg_run.pop("videos", None)
                cfg_run["out_root"] = str(run_out)
                # clips already start at the window; full videos must seek to it
                cfg_run["pipeline"]["start_frame"] = 0 if make_clips else int(w_start)

                t0 = time.time()
                metrics = run_one(
                    base_cfg=cfg_run,
                    run=run,
                    window_dir=Path(cfg_run["video_dir"]),
                    out_root=run_out,
                    max_frames_override=max_frames_override,
                )
                dt = time.time() - t0

                row = {
                    "variant": variant,
                    "window_start": w_start,
                    "window_len": w_len,
                    **metrics,
                }

                writer.writerow(row)
                all_rows.append(row)

                logging.info(
                    "[DONE] %s | %s | %s | window %d:%d | global_ids=%s | largest_comp=%s | %.1fs",
                    variant, experiment, run.tag, w_start, w_start + w_len,
                    row["global_ids"], row["largest_component_local_tracks"], dt
                )

                subset = [
                    r for r in all_rows
                    if r["variant"] == variant
                    and int(r["window_start"]) == int(w_start)
                    and int(r["window_len"]) == int(w_len)
                ]
                plots_dir = window_work / "plots"
                title_prefix = f"{variant} | {experiment} | window {w_start}:{w_start+w_len}"
                save_plots_for_window_variant(
                    rows=subset,
                    experiment=experiment,
                    out_dir=plots_dir,
                    title_prefix=title_prefix,
                )

    logging.info("Wrote: %s", summary_path)
    logging.info("Experiment output root: %s", study_out)


# --------------------------
# Main
# --------------------------

def parse_experiments(arg_experiments: str, arg_experiment: str) -> List[str]:
    """
    Backward compatible:
    - If --experiments provided -> use it
    - Else fall back to --experiment (single)
    """
    s = (arg_experiments or "").strip()
    if s:
        if s.lower() == "all":
            return ALL_EXPERIMENTS[:]
        exps = [e.strip() for e in s.split(",") if e.strip()]
        for e in exps:
            if e not in ALL_EXPERIMENTS:
                raise ValueError(f"Unknown experiment '{e}'. Valid: {', '.join(ALL_EXPERIMENTS)}")
        return exps

    one = (arg_experiment or "").strip()
    if not one:
        raise ValueError("Provide --experiment <name> OR --experiments <list|all>.")
    if one not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{one}'. Valid: {', '.join(ALL_EXPERIMENTS)}")
    return [one]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", action="append", default=[],
                    help="YAML file(s) layered over configs/default.yaml")
    ap.add_argument("-o", "--set", dest="overrides", action="append", default=[],
                    help="Config override, e.g. -o pose.type=alphapose")
    ap.add_argument("--pose", type=str, default="", help="Shortcut for -o pose.type=<name>")
    ap.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=VARIANTS + ["both", "all"],
        help="tracker_reid combination(s): both = deepsort_resnet + strongsort_osnet, all = all four.",
    )
    ap.add_argument("--keep-videos", action="store_true", help="Also write annotated videos per run.")

    # Backward compatible single experiment
    ap.add_argument("--experiment", type=str, default="", help="Single experiment (legacy).")

    # New: multi experiment
    ap.add_argument(
        "--experiments",
        type=str,
        default="",
        help='Comma-list of experiments or "all". Example: "stitch_cos_thr,tracker_ds_max_age" or "all"',
    )

    ap.add_argument(
        "--windows",
        type=str,
        default="",
        help='Windows "start:end,start:end,...". Example: "0:5000,30000:35000,80000:85000"',
    )
    ap.add_argument("--make-clips", action="store_true", help="ffmpeg trim into clips per window.")
    ap.add_argument("--max-frames", type=int, default=0, help="Override max frames processed (0=window_len).")
    ap.add_argument(
        "--study-out",
        type=str,
        default="",
        help="Override output dir (applies only for single experiment; for multi, use config out_root).",
    )
    ap.add_argument("--clean", action="store_true", help="Delete output dir(s) before running.")
    ap.add_argument("--osnet-weights", type=str, default="", help="Override OSNet weights path.")
    ap.add_argument("--resnet-weights", type=str, default="", help="Override ResNet weights path.")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    overrides = list(args.overrides) + ([f"pose.type={args.pose}"] if args.pose else [])
    base_cfg_raw = load_config(args.config, overrides)

    session_dir = Path(base_cfg_raw["video_dir"]).resolve()
    if not session_dir.exists():
        raise SystemExit(f"video_dir not found: {session_dir}")

    base_out = Path(base_cfg_raw["out_root"]).resolve()

    windows = parse_windows(args.windows)
    if not windows:
        mf = args.max_frames if args.max_frames > 0 else 5000
        windows = [(0, mf)]
        logging.info("No --windows provided. Using single window length=%d.", mf)

    experiments = parse_experiments(args.experiments, args.experiment)

    variants = {"both": ["deepsort_resnet", "strongsort_osnet"], "all": VARIANTS}.get(args.variant, [args.variant])
    osnet_w = args.osnet_weights.strip() or None
    resnet_w = args.resnet_weights.strip() or None

    # Run sequentially: variants x experiments
    for variant in variants:
        for experiment in experiments:
            # study-out override is only meaningful if you are running ONE experiment.
            # For multi-experiment mode, we keep the standard stable layout under out_root/study/.
            study_override = args.study_out.strip() if (len(experiments) == 1 and args.study_out.strip()) else None

            logging.info("============================================================")
            logging.info("RUNNING: variant=%s experiment=%s", variant, experiment)
            logging.info("============================================================")

            run_experiment(
                base_cfg_raw=base_cfg_raw,
                session_dir=session_dir,
                base_out=base_out,
                experiment=experiment,
                variant=variant,
                windows=windows,
                make_clips=args.make_clips,
                clean=args.clean,
                study_out_override=study_override,
                osnet_w=osnet_w,
                resnet_w=resnet_w,
                max_frames_cli=args.max_frames,
                keep_videos=args.keep_videos,
            )

    logging.info("ALL DONE.")


if __name__ == "__main__":
    main()
