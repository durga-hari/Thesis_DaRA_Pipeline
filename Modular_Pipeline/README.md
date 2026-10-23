# Modular Pipeline

Multi-camera person pose estimation, tracking and cross-camera re-identification
with interchangeable components. The Modular Pipeline merges `AlphaPose_OSNet_Pipeline` and
`RTMO_DeepSORT_Pipeline` into one codebase.

```
video ─► detector ─► pose estimator ─► ReID ─► tracker ─► results.jsonl   (per camera)
                                                              │
                         cross-camera stitching ◄─────────────┘
                         ─► crosscam_map.json, merged_global_tracks_union.jsonl, ID videos
```

| Stage | Options (`type:`) |
|---|---|
| detector | `yolov8` |
| pose | `alphapose` (HRNet-W32, COCO-17) · `rtmpose` (RTMPose-L WholeBody, 133 joints) |
| reid | `osnet` · `resnet50` |
| tracker | `deepsort` · `strongsort` |

Every stage uses the same interface regardless of backend, so you can combine any detector, pose model, ReID model and tracker.
The detector is a separate stage, so both pose backends see **identical boxes**.

## Layout

```
configs/default.yaml        all settings; one block per backend, `type:` selects
configs/experiments/        small overlays (alphapose_stack, rtmpose_stack, fair_comparison)
posepipe/
  detection/ pose/ reid/ tracking/   pluggable components + registries
  crosscam/                          stitching, global-ID merge, ID videos
  runner.py                          per-camera loop + multi-camera session
  evaluation/                        metrics, plots, parameter study (ported from RTMO)
  third_party/deep_sort/             vendored DeepSORT core (numpy/scipy only)
externals/AlphaPose/        vendored AlphaPose (pure-PyTorch parts used; no build step)
checkpoints/                weights (see checkpoints/README.md)
tests/                      end-to-end tests, no GPU or weights needed
```

## Install

```bash
conda env create -f environment.yml
conda activate posepipe
mim install "mmengine==0.10.4" "mmcv==2.1.0" "mmpose==1.3.2"   # only for pose.type=rtmpose
```

Then download the weights listed in [checkpoints/README.md](checkpoints/README.md).
Each backend is imported only when you select it. AlphaPose runs without MMPose
installed, and RTMPose runs without AlphaPose being importable.

**GPU notes.** The pinned torch 2.1.2/cu121 matches MMCV 2.1.0's prebuilt wheels.
RTX 50-series GPUs (sm_120) need CUDA 12.8 and torch ≥ 2.7. In that case install torch for cu128 and
build MMCV 2.1.0 from source with `MMCV_WITH_OPS=1`, as the old `RTMO_DeepSORT_Pipeline/setup_env.sh` did.
AlphaPose, YOLO and torchreid have no such constraint.

## Run

Put one video per camera in `data/videos/` (or set `video_dir` / `videos`).

```bash
python -m posepipe list                                    # registered components
python -m posepipe show -o pose.type=alphapose             # resolved config, nothing runs

python -m posepipe run                                     # defaults: rtmpose + resnet50 + deepsort
python -m posepipe run -o pose.type=alphapose              # toggle the pose backend
python -m posepipe run -o pose.type=alphapose -o reid.type=osnet -o tracker.type=strongsort
python -m posepipe run -c configs/experiments/alphapose_stack.yaml   # old AlphaPose stack
python -m posepipe run -c configs/experiments/rtmpose_stack.yaml     # old RTMO stack
```

Outputs go to `outputs/<pose>_<reid>_<tracker>/`, so different combinations never
overwrite each other. Each run folder contains `run_config.yaml`, the exact config used.

```
outputs/alphapose_osnet_strongsort/
  run_config.yaml  summary_videos.csv  crosscam_map.json  crosscam_summary.csv
  merged_global_tracks_union.jsonl
  <cam>_output/ results.jsonl  annotated.mp4  <cam>_output_poses_union.jsonl
                <cam>_output_localids.mp4  <cam>_output_globalids.mp4
```

`results.jsonl` has one line per tracked person per frame:
`frame, id, bbox (tracker), det_bbox, keypoints [[x,y,conf]...], feature, score, pose_conf, det_score`.

## Evaluate and compare

```bash
python -m posepipe evaluate --root outputs/alphapose_osnet_deepsort   # per_video.csv, overall.csv, ...
python -m posepipe plots    --root outputs/alphapose_osnet_deepsort
python -m posepipe stitch   --root outputs/alphapose_osnet_deepsort -o crosscam.cos_thr=0.5   # re-stitch only

# parameter study, now for any pose backend
python -m posepipe study --pose alphapose --variant all --experiments all --windows 0:5000
```

**For a like-for-like pose comparison**, use `configs/experiments/fair_comparison.yaml`.
It gives both backends the same detector boxes, the same 17 body joints
(`keypoint_format: coco17`; COCO-WholeBody joints 0–16 *are* COCO-17), and the same ReID model and tracker:

```bash
python -m posepipe run -c configs/experiments/fair_comparison.yaml -o pose.type=alphapose
python -m posepipe run -c configs/experiments/fair_comparison.yaml -o pose.type=rtmpose
```

Joint confidences are not calibrated across models: HRNet heatmap peaks and RTMPose
SimCC scores mean different things. A shared `min_mean_kp_conf` threshold therefore filters the two
backends differently, so report or tune it per backend.

## Adding a component

Write a class with the stage's interface (documented in each stage's `__init__.py`), then register it:

```python
# posepipe/pose/__init__.py
POSE.register_lazy("vitpose", "posepipe.pose.vitpose:ViTPoseEstimator")
```

Give it a settings block in `configs/default.yaml` (`pose: {vitpose: {...}}`) and select it with `-o pose.type=vitpose`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests run the real runner, trackers, stitching and merge on synthetic two-camera
videos. A randomly initialised HRNet exercises the AlphaPose pre- and post-processing.
They need no GPU and no downloaded weights.

## Differences from the two original repositories

Some of these fixes change results compared with earlier runs:

1. **Keypoints matched to the wrong track (RTMO + DeepSORT).** `tracks[i]` was paired
   with `det_kps[i]`, but DeepSORT does not return tracks in detection order, so
   skeletons, confidences and features could belong to a different person. Trackers now
   return `det_index`, carried through DeepSORT's `Detection.mask`.
2. **Global IDs injected onto the wrong local tracks.** `crosscam_stitch.py` keyed the
   map by merged-group index (`"0","1",…`) while `inject_global_ids` looked up real
   local track IDs. The map is now keyed by the real IDs.
3. **Intra-camera merge settings were ignored.** `intra_merge_*` in the config and the
   parameter study never reached the stitcher, which used fixed values (0.85 similarity, 30 frames).
   They are now applied. Defaults reproduce the old effective behaviour (`emb_thr 0.15`, `time_gap 30`, `iou_thr 0`).
4. **AlphaPose preprocessing.** The wrapper divided by an ImageNet std that
   AlphaPose's own `test_transform` does not use (inputs about 4.4× too large), used a square crop instead of
   the aspect-preserving one, and decoded joints by plain argmax. It now uses AlphaPose's own
   transform and `heatmap_to_coord_simple`.
5. **RTMPose colour order.** MMPose expects BGR and converts internally; the old code passed RGB.
   `pose.rtmpose.input_color: rgb` reproduces the old behaviour.
6. **DeepSORT on empty frames** now calls `update([])`, so missed tracks age and get
   deleted on schedule.
7. **Parameter study.** Windows without `--make-clips` now seek to the window start
   (previously every window started at frame 0). Per-run videos are off by default. Outputs are split by pose backend.
8. Annotated ID videos were drawn one frame out of step; this is fixed.
9. The old AlphaPose pipeline's `--detector alphapose` (YOLOv3) path called a `detect`
   method that `AlphaPoseRunner` did not have, so it never ran. It was not carried over; YOLOv8 is the detector.

The `strongsort` tracker is the lightweight version from the RTMO repo: greedy IoU-gated
cosine matching with a Kalman filter. It does not implement the full StrongSORT method (Hungarian matching,
EMA features, camera-motion compensation), so describe it accordingly in write-ups.
