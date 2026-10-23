# Thesis DaRA Pipeline

**Development of a Robust Pipeline for Human Skeleton Generation From Multi-Perspective Camera Data in the DaRA Dataset**

Master's thesis by Durga Shanmukha Sridatta Arun Hari Haranath. Study programme: Automation and Robotics.
Technische Universität Dortmund, Fakultät Maschinenbau, [Lehrstuhl für Förder- und Lagerwesen](http://flw.mb.tu-dortmund.de/).
Supervisors: Prof. Dr.-Ing. Alice Kirchheim and Friedrich Niemann, M.Sc.
Submitted 13 January 2026.

This repository contains the **modular end-to-end pipeline** developed in the thesis. It takes raw
multi-camera RGB video from a warehouse and produces, for every person, a 2D skeleton,
an appearance embedding, a temporally consistent track per camera and a global identity across cameras.

---

## Contents

- [Motivation](#motivation)
- [The DaRA dataset](#the-dara-dataset)
- [Objectives](#objectives)
- [Pipeline architecture](#pipeline-architecture)
- [Components](#components)
- [Unsupervised validation metrics](#unsupervised-validation-metrics)
- [Experiments and results](#experiments-and-results)
- [Key findings](#key-findings)
- [Getting started](#getting-started)
- [Reproducing the thesis configurations](#reproducing-the-thesis-configurations)
- [Repository layout](#repository-layout)
- [Future work](#future-work)
- [Citation](#citation)
- [Third-party code and licences](#third-party-code-and-licences)

---

## Motivation

Warehouses increasingly have mobile industrial robots and AGVs working alongside people. Safety
monitoring, ergonomic assessment and human–robot collaboration all depend on reliable perception
of human posture and motion. Pose estimators are usually trained and benchmarked on COCO, MPII or
PoseTrack, which mostly show near-frontal views, little occlusion and short clips. Warehouse footage
is different: cameras are mounted high, people are frequently occluded while picking and carrying,
lighting is uncontrolled, and recordings run for hours.

Pose estimation is also usually studied in isolation. In a real deployment, pose quality depends on
how well people are detected, how stably they are tracked and how consistently they are identified
across cameras, and errors made early propagate downstream. This thesis therefore builds and
evaluates the **whole chain**, from detection to cross-camera identity, on real warehouse data.

## The DaRA dataset

DaRA is a new dataset recorded in a warehouse and logistics setting.

- **Size:** 109 hours of video.
- **Cameras:** recorded simultaneously by **3 ego-view action cameras** (GoPro) and **6 fixed third-person cameras**.
- **Labels:** fully **unannotated**. There are no ground-truth keypoints, bounding boxes or identity labels.
- **Sessions:** footage is organised in sessions of nine synchronised videos, about 1 h 10 min each. All experiments use **Session 4**.

Videos are renamed before processing (thesis Table 3.1):

| Short name | Camera |
|---|---|
| `AC10`, `AC11`, `AC12` | Action cameras (ego view) of subjects 10, 11 and 12 |
| `FC1` | Fixed camera facing the goods issue area |
| `FC2` | Fixed camera facing the packing area |
| `FC3` | Fixed camera facing rack complex 1 |
| `FC4` | Fixed camera in the office |
| `FC5` | Fixed camera, overview 1: rack storage system |
| `FC6` | Fixed camera, overview 2: open areas |

Without annotations, the standard supervised metrics (AP/mAP, PCK/PCKh, MOTA/MOTP, IDF1) can't be
computed. The pipeline is instead assessed with **unsupervised proxy metrics**
([see below](#unsupervised-validation-metrics)) and visual inspection of annotated videos.

## Objectives

1. **Modularity.** Components can be replaced or extended with minimal restructuring.
2. **Multi-person, multi-camera robustness.** Track multiple actors across simultaneous camera views.
3. **Efficiency.** Near real-time processing without excessive computational overhead.
4. **Occlusion handling.** Cope with visually complex, crowded warehouse scenes.

## Pipeline architecture

The pipeline has three stages (thesis Chapter 3):

```
 PRE-PROCESSING            PROCESSING (frame-local, no temporal reasoning)            POST-PROCESSING
┌────────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────────┐   ┌────────────────────────┐
│ rename videos  │   │  Detection   │   │ Pose estimation  │   │ Appearance feature│   │ Temporal association │   │ Cross-camera identity  │
│ decode frames  │──►│  YOLOv8m     │──►│ AlphaPose /      │──►│ extraction        │──►│ (per camera)         │──►│ consolidation          │
│ frame index,   │   │  person boxes│   │ RTMPose          │   │ ResNet-50 / OSNet │   │ DeepSORT / StrongSORT│   │ hybrid ReID + pose sim │
│ H×W, fps       │   │  + scores    │   │ (x, y, conf) × J │   │ L2-normalised f∈R^D│  │ local track IDs      │   │ mutual match, union-find│
└────────────────┘   └──────────────┘   └──────────────────┘   └───────────────────┘   └──────────────────────┘   └────────────────────────┘
                                                                                                                              │
                                     JSON per camera + merged global JSON + annotated videos (local / global IDs) ◄───────────┘
```

1. **Pre-processing.** Videos are decoded sequentially at their native frame rate, with no frame
   skipping or interpolation. Each frame carries its frame index, which anchors everything downstream in time,
   plus its resolution. The frame rate and frame count are kept as context.
2. **Processing.** Each frame is handled independently:
   - **Detection:** person bounding boxes with confidence scores, `B_t = {(x1, y1, x2, y2, s)}`.
   - **Pose estimation:** for every box, `P_i = {(x_j, y_j, c_j)}` over the J keypoints of the body model, plus a mean pose confidence.
   - **Appearance features:** an L2-normalised ReID embedding `f_i ∈ R^D` from the person crop.
3. **Post-processing.**
   - **Temporal association:** per-camera multi-object tracking assigns local track IDs.
   - **Cross-camera consolidation:** each track's embeddings and poses are averaged into one track-level
     descriptor. Fragments of the same person within a camera are merged first. A hybrid similarity is then
     computed between tracks in different cameras: `s_hyb = α·s_app + (1−α)·s_pose`.
     Mutual best matches above a threshold are joined with **union-find**, and each connected component
     becomes one **global ID**.
   - **Output:** for each camera, a JSON record per person per frame (box, keypoints, embedding, local ID and global ID),
     plus annotated videos. Skeletons are colour-coded by ID.

## Components

Every component can be swapped in or out from the configuration (`type:` in `configs/default.yaml`, or `-o stage.type=…` on the command line).

| Stage | Component | Model / weights | Key properties |
|---|---|---|---|
| Detection | **YOLOv8m** | Ultralytics `yolov8m.pt`, COCO, frozen | single-stage, anchor-free, decoupled head; ~25.9 M params; 640×640 |
| Pose | **AlphaPose** | HRNet-W32, 256×192, COCO-17 | top-down heatmap regression; 64×48 heatmaps; ~28.5 M params |
| Pose | **RTMPose-L** (MMPose) | CSPNeXt-L, 384×288, COCO-WholeBody-133 | top-down SimCC coordinate classification; ~26 M params |
| ReID | **ResNet-50** | Market-1501 (751 IDs) | 2048-d embedding; ~25.6 M params |
| ReID | **OSNet-x1.0** | MSMT17 (1,041 IDs) | omni-scale, 512-d embedding; ~2.2 M params |
| Tracking | **DeepSORT** | n/a | Kalman filter (x, y, a, h), Mahalanobis gating, cosine feature gallery, matching cascade + Hungarian |
| Tracking | **StrongSORT-style** | n/a | Kalman filter, IoU gate, cosine appearance matching, time-to-live |
| Cross-camera | **Hybrid union-find stitching** | n/a | intra-camera fragment merge, `α`-weighted ReID + pose similarity, mutual best match |

The thesis evaluates two main stacks: **ResNet-50 + DeepSORT** (baseline) and **OSNet + StrongSORT**
(more robust to viewpoint change and partial occlusion). The modular design allows any combination.

**Notes on what this repository implements:**
- **StrongSORT.** Thesis §4.4.2 describes StrongSORT as published: an EMA appearance prototype,
  confidence-scaled Kalman noise and a unified cost function. The tracker here is a lightweight
  StrongSORT-*style* implementation: Kalman filter, IoU gate, greedy cosine matching and track TTL.
- **Halpe-136.** The thesis also evaluates AlphaPose with the Halpe-136 whole-body model
  (FastPose ResNet-50). That body model is **not yet a selectable option** in `Modular_Pipeline`.
  Currently it supports AlphaPose HRNet COCO-17 and RTMPose COCO-WholeBody-133.

## Unsupervised validation metrics

Since DaRA has no labels, each stage is assessed with proxy metrics for stability and plausibility
(thesis §4.5). `python -m posepipe evaluate --root <run_dir>` writes them to `per_video.csv`,
`overall.csv` and `crosscam.csv`:

| Stage | Metric (thesis) | Definition | Output column |
|---|---|---|---|
| Detection | Detection stability `CV_det` | σ(N_t) / μ(N_t) of persons per frame | `frame_detection_stability` |
| Detection | Temporal persistence | mean best IoU of boxes between consecutive frames | `temporal_persistence_rate` |
| Pose | Pose jitter | mean per-joint displacement between consecutive frames within a track | `mean_pose_jitter` |
| Pose | Limb-length consistency | coefficient of variation of bone lengths over a track (limbs are rigid) | `mean_limb_length_cv_over_tracks` |
| Tracking | Track fragmentation index | share of short tracks (< 10 frames) among all tracks | `track_fragmentation_index` |
| ReID | Embedding temporal variance | spread of a track's embeddings over time | `mean_embedding_temporal_std` |
| ReID | Proxy identity switches | IoU(b_t, b_t+1) < 0.3 **and** ‖f_t+1 − f_t‖ > 0.7 | `proxy_id_switches`, `proxy_id_checks` |
| Cross-camera | Global identity count / graph connectivity | number of global IDs and how connected the identity graph is | `global_id_count`, `graph_connectivity_ratio` |

`python -m posepipe plots --root <run_dir>` draws the figures used in Chapter 5:
- detection coverage vs. persistence
- limb-length CV distributions
- track-length histograms
- camera participation per global ID
- per-camera time series

## Experiments and results

All experiments use DaRA **Session 4**, processing up to **50,000 frames per camera**, with YOLOv8m as the fixed detector.

### Parameter sensitivity (5,000 frames per camera)

| Parameter | Range | Effect |
|---|---|---|
| Cross-camera cosine threshold | 0.35 → 0.85 | **Most influential.** Global IDs rise from 4 to 37 (OSNet). Low thresholds over-merge, high thresholds fragment |
| DeepSORT max cosine distance | 0.15 → 0.30 | Global IDs 8 → 15 and local tracks 417 → 488 (ResNet-50) |
| DeepSORT max age / StrongSORT TTL | 50 → 250 frames | Mainly changes the number of short local fragments; global IDs stay at 9 |
| StrongSORT similarity threshold | 0.35 → 0.65 | Little effect (9 global IDs throughout) |
| Cross-camera temporal slack | 0 → 300 frames | Secondary effect (9 → 10 global IDs) |

These trends were the same for both pose backends. They are driven by the appearance embeddings and
association thresholds, not by the pose estimator.

### Pipeline configurations (thesis Tables 5.9–5.13)

| Configuration | Skeleton | Pose jitter ↓ | Limb-length CV ↓ | Fragmentation ↓ | Proxy ID switches | Global IDs | Graph connectivity |
|---|---|---:|---:|---:|---:|---:|---:|
| AlphaPose + ResNet-50 + DeepSORT | COCO-17 | 465.85 | 0.287 | 0.554 | 807,287 | 1 | 1.000 |
| AlphaPose + OSNet + StrongSORT | COCO-17 | 451.03 | 0.274 | 0.718 | 978,018 | 33 | 0.212 |
| AlphaPose + OSNet + StrongSORT | Halpe-136 | 487.20 | 0.312 | 0.762 | 1,009,412 | 35 | 0.186 |
| RTMPose + ResNet-50 + DeepSORT | COCO-WB-133 | 111.81 | 0.480 | 0.000 | 471 | 418 | 0.151 |
| RTMPose + OSNet + StrongSORT | COCO-WB-133 | **35.03** | 0.372 | 0.000 | **107** | 259 | 0.070 |

Detection stability (CV_det ≈ 0.45–0.47) and temporal persistence (0.97–0.99) were essentially the same
in every configuration, which confirms that the detector behaves the same regardless of pose backend.

> **Provenance.** These numbers come from the original prototypes (`AlphaPose_OSNet_Pipeline` and
> `RTMO_DeepSORT_Pipeline`). `Modular_Pipeline` merges those prototypes and corrects several
> implementation issues found while merging. These include how keypoints are paired with tracks, how global IDs
> are mapped back to local tracks, the AlphaPose preprocessing and the colour order fed to RTMPose
> (see [`Modular_Pipeline/README.md`](Modular_Pipeline/README.md#differences-from-the-two-original-repositories)).
> Re-running the configurations here can therefore give different values.

## Key findings

- **Identity association is the main failure mode.** Once detection and pose estimation are fixed, behaviour
  across all configurations is governed by appearance-embedding discrimination and association thresholds.
  ResNet-50 + DeepSORT under AlphaPose over-merges everyone into a single global identity. OSNet + StrongSORT
  avoids that over-merging but fragments identities instead.
- **Better poses alone do not fix cross-camera identity.** RTMPose WholeBody cuts pose jitter by roughly 4–13×
  compared with AlphaPose COCO-17, yet global identities are still inflated across non-overlapping cameras.
  Pose quality is *necessary but not sufficient*.
- **More keypoints are not automatically better under AlphaPose.** The Halpe-136 model increased jitter, limb-length
  variability and fragmentation compared with COCO-17. COCO-17 remains the practical choice for AlphaPose.
- **Viewpoint matters.** Fixed overhead cameras give stable poses and long tracks. Ego-centric action
  cameras show higher jitter and more fragmentation, because of camera motion, truncation and blur that
  models trained on standard datasets don't handle.
- **Modularity achieved.** Multiple pose backbones and identity stacks were integrated and evaluated without
  changing the overall pipeline structure.

## Getting started

```bash
git clone https://github.com/durga-hari/Thesis_DaRA_Pipeline.git
cd Thesis_DaRA_Pipeline/Modular_Pipeline

conda env create -f environment.yml
conda activate posepipe
mim install "mmengine==0.10.4" "mmcv==2.1.0" "mmpose==1.3.2"   # only needed for RTMPose
```

1. Download the model weights listed in [`Modular_Pipeline/checkpoints/README.md`](Modular_Pipeline/checkpoints/README.md). Weights are not stored in this repository.
2. Put one video per camera in `Modular_Pipeline/data/videos/`, using the short names above (`AC10.mp4`, `FC1.mp4`, …).
3. Run and evaluate:

```bash
python -m posepipe list                                    # available components
python -m posepipe run -o pose.type=rtmpose -o reid.type=osnet -o tracker.type=strongsort
python -m posepipe evaluate --root outputs/rtmpose_osnet_strongsort
python -m posepipe plots    --root outputs/rtmpose_osnet_strongsort
```

Each configuration writes to its own folder, `outputs/<pose>_<reid>_<tracker>/`. The folder contains:
- `<cam>_output/results.jsonl` for each camera
- `crosscam_map.json` and `merged_global_tracks_union.jsonl`
- videos annotated with local and global IDs
- `run_config.yaml`, the exact configuration used

## Reproducing the thesis configurations

Run from `Modular_Pipeline/`. `-o pipeline.max_frames=50000` matches the thesis frame limit.

| Thesis configuration | Command |
|---|---|
| AlphaPose + ResNet-50 + DeepSORT (COCO-17) | `python -m posepipe run -o pose.type=alphapose -o reid.type=resnet50 -o tracker.type=deepsort -o pipeline.max_frames=50000` |
| AlphaPose + OSNet + StrongSORT (COCO-17) | `python -m posepipe run -o pose.type=alphapose -o reid.type=osnet -o tracker.type=strongsort -o pipeline.max_frames=50000` |
| RTMPose + ResNet-50 + DeepSORT (COCO-WB) | `python -m posepipe run -o pose.type=rtmpose -o reid.type=resnet50 -o tracker.type=deepsort -o pipeline.max_frames=50000` |
| RTMPose + OSNet + StrongSORT (COCO-WB) | `python -m posepipe run -o pose.type=rtmpose -o reid.type=osnet -o tracker.type=strongsort -o pipeline.max_frames=50000` |
| Parameter sensitivity study (5,000 frames) | `python -m posepipe study --pose alphapose --variant both --experiments all --windows 0:5000` (repeat with `--pose rtmpose`) |

To compare both pose backends on identical joints, layer `-c configs/experiments/fair_comparison.yaml`
on top. It restricts both backends to the 17 COCO body joints. All options are documented in
[`Modular_Pipeline/README.md`](Modular_Pipeline/README.md).

## Repository layout

```
Modular_Pipeline/
  configs/        default.yaml (every option; `type:` selects each component) and experiment presets
  posepipe/
    detection/    YOLOv8 person detector
    pose/         AlphaPose (HRNet) and RTMPose backends
    reid/         OSNet / ResNet-50 appearance embeddings (torchreid)
    tracking/     DeepSORT and StrongSORT-style trackers
    crosscam/     intra-camera merge, hybrid union-find stitching, global-ID merge, ID videos
    evaluation/   unsupervised metrics, plots, parameter study
    runner.py     per-camera processing and multi-camera session
  externals/      vendored AlphaPose (pure-PyTorch parts used; no build step)
  checkpoints/    model weights (not tracked)
  tests/          end-to-end tests, run without a GPU or weights
```

Tests: `cd Modular_Pipeline && python -m unittest discover -s tests -v`

## Future work

From thesis Chapter 6:
- **Identity association:** add longer temporal context, motion cues and spatial constraints from the
  warehouse layout to cross-camera matching. Replace threshold-based stitching with learned association models.
- **Ego-centric robustness:** camera-motion compensation, temporal pose filtering, or pose models trained on ego-centric data.
- **Evaluation:** a partially annotated DaRA subset, to enable supervised validation and calibrate the unsupervised metrics.
- **Deployment:** fully online operation, and integration with downstream warehouse safety or activity-recognition systems.

## Citation

```bibtex
@mastersthesis{HariHaranath2026DaRA,
  author = {Hari Haranath, Durga Shanmukha Sridatta Arun},
  title  = {Development of a Robust Pipeline for Human Skeleton Generation From
            Multi-Perspective Camera Data in the {DaRA} Dataset},
  school = {Technische Universit{\"a}t Dortmund, Lehrstuhl f{\"u}r F{\"o}rder- und Lagerwesen},
  type   = {Master's thesis},
  year   = {2026},
  month  = jan
}
```

## Third-party code and licences

| Component | Source | Licence |
|---|---|---|
| AlphaPose (vendored in `externals/AlphaPose`) | [MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) | [Licence](Modular_Pipeline/externals/AlphaPose/LICENSE): academic / non-commercial research use only |
| DeepSORT core (vendored in `posepipe/third_party/deep_sort`) | [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch), based on [nwojke/deep_sort](https://github.com/nwojke/deep_sort) | [Licence](Modular_Pipeline/posepipe/third_party/deep_sort/LICENSE) |
| MMPose / RTMPose | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | Apache-2.0 (installed via pip) |
| Ultralytics YOLOv8 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 (installed via pip) |
| torchreid | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | MIT (installed via pip) |

Model weights belong to their respective authors and are downloaded separately. The DaRA dataset is not included in this repository.
