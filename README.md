# Multi-Camera Human Pose Tracking for Warehouse Robotics

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Tests](https://img.shields.io/badge/tests-unittest-brightgreen)

modular end-to-end pipeline capable of extracting temporally consistent human skeletons from
raw multi-camera video data recorded in real-world warehouse environments per person, per frame, and consistent across varied camera views.
It's built for settings where robots and AGVs share space with people. Safety monitoring,
ergonomics and human–robot collaboration all need to know *where each person is and how they move*.

Developed as my Master's thesis in Automation and Robotics at **TU Dortmund University**
(Chair of Materials Handling and Warehousing). It runs on the **DaRA** warehouse dataset:
nine synchronised cameras, six fixed and three worn by workers, with no annotations.

---

## Highlights

- **End-to-end system.** Raw video in, then detection → pose estimation → appearance
  embeddings → multi-object tracking → cross-camera identity. Out come per-camera tracks, global
  identities and annotated videos.
- **Plug-and-play architecture.** Each stage sits behind a common interface and a component
  registry. Swapping AlphaPose for RTMPose, or DeepSORT for StrongSORT, is a one-word change in config or on the command line.
- **Multi-camera identity.** Track-level appearance and pose descriptors, intra-camera fragment
  merging, and mutual-best-match **union-find** clustering that assigns one global ID per person across views.
- **Evaluation without ground truth.** A suite of unsupervised metrics for detection stability, pose jitter,
  limb-length consistency, track fragmentation, identity switches and identity-graph structure, for data that has no labels.
- **Experiment tooling.** Layered YAML configs, CLI overrides, automatic per-configuration
  output folders with the exact config saved alongside, and a parameter-sweep runner with plots.
- **Tested without a GPU.** End-to-end tests run the real tracking, stitching and evaluation
  code on synthetic two-camera video, so it runs on any laptop or CI runner.

## Architecture

```
            per camera (frame-local)                                       per camera (temporal)      across cameras
┌──────────┐   ┌────────────────────┐   ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
│ Detector │──►│ Pose estimator     │──►│ Appearance ReID   │──►│ Multi-object tracker │──►│ Cross-camera stitching   │
│ YOLOv8   │   │ AlphaPose (HRNet)  │   │ OSNet / ResNet-50 │   │ DeepSORT /           │   │ hybrid ReID + pose sim,  │
│          │   │ RTMPose (WholeBody)│   │ L2-norm embedding │   │ StrongSORT-style     │   │ union-find → global IDs  │
└──────────┘   └────────────────────┘   └───────────────────┘   └──────────────────────┘   └──────────────────────────┘
      boxes + scores    (x, y, conf) × 17 / 133 joints     f ∈ R^512 / R^2048          local track IDs        JSONL + annotated videos
```

| Stage | Interchangeable components |
|---|---|
| Detection | YOLOv8 (m / s weights) |
| Pose estimation | **AlphaPose** HRNet-W32: heatmap regression, 17 COCO joints · **RTMPose-L**: SimCC coordinate classification, 133 whole-body joints |
| Re-identification | **OSNet-x1.0** (MSMT17) · **ResNet-50** (Market-1501) |
| Tracking | **DeepSORT**: Kalman filter, Mahalanobis gating, matching cascade · **StrongSORT-style**: Kalman filter, IoU gate, cosine matching |
| Cross-camera | Intra-camera fragment merge, α-weighted appearance + pose similarity, mutual best match, union-find |

## Engineering design

**Component registry with lazy loading.** Backends are registered by name and imported only
when selected. An AlphaPose run never imports MMPose, and an RTMPose run never needs AlphaPose built.

```python
POSE.register_lazy("alphapose", "posepipe.pose.alphapose_hrnet:AlphaPoseEstimator")
POSE.register_lazy("rtmpose",   "posepipe.pose.rtmpose:RTMPoseEstimator")

est = POSE.build({"type": "rtmpose", "cfg": ..., "ckpt": ...})
results = est(frame_bgr, boxes)   # same interface for every backend
```

**One config, one toggle per stage.** Each stage keeps a settings block for every backend, and
`type:` selects which one is used, so switching never leaves stale parameters behind:

```yaml
pose:
  type: rtmpose            # ← the toggle
  rtmpose:   {cfg: ..., ckpt: ...}
  alphapose: {cfg: ..., ckpt: ..., batch_size: 64}
```

```bash
python -m posepipe run -o pose.type=alphapose -o reid.type=osnet -o tracker.type=strongsort
```

**Correctness by construction.**
- Trackers return the index of the detection each track matched, so skeletons and embeddings
  always stay attached to the right person.
- Each pose backend uses its framework's own preprocessing and decoding.
- Comparisons between backends can use the same detector boxes and the same 17-joint subset.

**Reproducible experiments.**
- Every run writes to `outputs/<pose>_<reid>_<tracker>/` together with the resolved `run_config.yaml`.
- Models are cached across parameter sweeps.
- Relative paths make configs portable between machines.

**Extensible.** Adding a component, for example ViTPose or ByteTrack, means one class with the
stage's interface, one registry line and one config block. The pipeline itself doesn't change.

## Evaluation without labels

DaRA has no ground-truth keypoints, boxes or identities. Standard metrics (AP, PCK, MOTA, IDF1)
can't be computed. The pipeline instead measures internal consistency and physical plausibility:

| Stage | Metrics |
|---|---|
| Detection | detection-count stability (coefficient of variation), frame-to-frame IoU persistence |
| Pose | per-joint jitter, limb-length coefficient of variation (bones are rigid) |
| Tracking | fragmentation index, track-length distribution |
| ReID | embedding temporal spread, proxy identity switches (box jump + embedding jump) |
| Cross-camera | global identity count, identity-graph connectivity, cameras per identity |

```bash
python -m posepipe evaluate --root outputs/rtmpose_osnet_strongsort   # CSV tables
python -m posepipe plots    --root outputs/rtmpose_osnet_strongsort   # figures
python -m posepipe study --pose rtmpose --variant all --experiments all --windows 0:5000   # parameter sweeps
```

## What the experiments showed

- **Identity association is the bottleneck, not pose quality.** Dense whole-body skeletons from
  RTMPose gave smoother, more stable poses than AlphaPose COCO-17. Cross-camera identity, however, was governed
  by appearance-embedding discrimination and association thresholds. That points future work at learned
  association rather than bigger pose models.
- **The cross-camera similarity threshold is the most sensitive parameter.** Sweeping it moves the system from
  over-merging, where different people share one ID, to fragmentation, where one person gets many IDs.
- **Viewpoint matters.** Fixed overhead cameras give stable skeletons and long tracks. Worker-worn ego
  cameras introduce motion blur and truncation, which models trained on standard datasets don't handle well.

## Tech stack

Python · PyTorch · OpenCV · NumPy / SciPy · pandas · Matplotlib · Ultralytics YOLOv8 · MMPose / RTMPose ·
AlphaPose · torchreid · Kalman filtering · Hungarian assignment · union-find · YAML-driven CLI · unittest

## Quick start

```bash
git clone https://github.com/durga-hari/Thesis_DaRA_Pipeline.git
cd Thesis_DaRA_Pipeline/Modular_Pipeline

conda env create -f environment.yml
conda activate posepipe
mim install "mmengine==0.10.4" "mmcv==2.1.0" "mmpose==1.3.2"   # only needed for RTMPose

python -m posepipe list                    # available components
python -m unittest discover -s tests -v    # no GPU or weights needed
```

To run on video:
1. Download the weights listed in [`Modular_Pipeline/checkpoints/README.md`](Modular_Pipeline/checkpoints/README.md).
2. Put one video per camera in `Modular_Pipeline/data/videos/`.
3. Run `python -m posepipe run`.

The full option reference is in [`Modular_Pipeline/README.md`](Modular_Pipeline/README.md).

## Repository layout

```
Modular_Pipeline/
  configs/        default.yaml (every option) and experiment presets
  posepipe/
    detection/    YOLOv8 person detector
    pose/         AlphaPose and RTMPose backends
    reid/         OSNet / ResNet-50 embeddings
    tracking/     DeepSORT and StrongSORT-style trackers
    crosscam/     fragment merging, union-find stitching, global-ID merge, ID-annotated videos
    evaluation/   unsupervised metrics, plots, parameter study
    registry.py   component registry with lazy imports
    runner.py     per-camera loop and multi-camera session
  externals/      vendored AlphaPose (pure-PyTorch parts; no build step)
  tests/          end-to-end tests on synthetic multi-camera video
```

## Thesis

*Development of a Robust Pipeline for Human Skeleton Generation From Multi-Perspective Camera Data in the DaRA Dataset*,
Master's thesis, TU Dortmund University, Chair of Materials Handling and Warehousing (FLW), 2026.
Supervised by Prof. Dr.-Ing. Alice Kirchheim and Friedrich Niemann, M.Sc.

## Third-party code and licences

AlphaPose (vendored, [academic / non-commercial licence](Modular_Pipeline/externals/AlphaPose/LICENSE)) ·
DeepSORT core (vendored, [licence](Modular_Pipeline/posepipe/third_party/deep_sort/LICENSE)) ·
MMPose (Apache-2.0) · Ultralytics YOLOv8 (AGPL-3.0) · torchreid (MIT).
Model weights and the DaRA dataset aren't included.
