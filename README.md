# Thesis DaRA Pipeline

Code for my Master's thesis: a **modular pipeline** for multi-camera human pose
estimation, multi-object tracking and cross-camera person re-identification.

Each stage of the pipeline can be swapped from the configuration. Different pose
estimators, ReID models and trackers can then be compared on the same videos,
with the same detections and the same evaluation.

```
 per camera                                                         across cameras
┌──────────┐   ┌───────────────┐   ┌────────────┐   ┌──────────┐   ┌──────────────────────┐
│ Detector │──►│ Pose estimator│──►│ ReID model │──►│ Tracker  │──►│ Cross-camera stitching│──► global IDs
└──────────┘   └───────────────┘   └────────────┘   └──────────┘   └──────────────────────┘
   YOLOv8        AlphaPose (HRNet)     OSNet           DeepSORT       ReID + pose similarity,
                 RTMPose (WholeBody)   ResNet-50       StrongSORT*    mutual match, union-find
```

## Components

| Stage | Options | Notes |
|---|---|---|
| Detector | YOLOv8 (`yolov8m`, `yolov8s`) | Separate from the pose model, so every pose backend gets identical boxes |
| Pose estimation | **AlphaPose** HRNet-W32 256×192 (COCO, 17 joints)<br>**RTMPose**-L 384×288 (COCO-WholeBody, 133 joints) | Top-down; either backend can be cut to the 17 COCO body joints for comparison |
| Re-identification | **OSNet**-x1.0 (MSMT17)<br>**ResNet-50** (Market-1501) | torchreid backbones, L2-normalised embeddings |
| Tracking | **DeepSORT**<br>**StrongSORT**\* | One tracker per camera |
| Cross-camera | Hybrid ReID + pose similarity, intra-camera fragment merge, union-find | Produces one global ID per person across cameras |

\* A lightweight StrongSORT-style tracker (Kalman filter, IoU gate, greedy cosine matching), not the full StrongSORT method.

## Quick start

```bash
git clone https://github.com/durga-hari/Thesis_DaRA_Pipeline.git
cd Thesis_DaRA_Pipeline/Modular_Pipeline

conda env create -f environment.yml
conda activate posepipe
mim install "mmengine==0.10.4" "mmcv==2.1.0" "mmpose==1.3.2"   # only needed for RTMPose
```

1. Download the model weights listed in [`Modular_Pipeline/checkpoints/README.md`](Modular_Pipeline/checkpoints/README.md). Weights are not stored in this repository.
2. Put one video per camera in `Modular_Pipeline/data/videos/`.
3. Run:

```bash
python -m posepipe list                                   # available components
python -m posepipe run -o pose.type=alphapose             # AlphaPose + defaults
python -m posepipe run -o pose.type=rtmpose -o reid.type=osnet -o tracker.type=strongsort
python -m posepipe evaluate --root outputs/rtmpose_osnet_strongsort
```

Each combination writes to its own folder, `outputs/<pose>_<reid>_<tracker>/`. The folder holds
the per-camera tracks (`results.jsonl`), the cross-camera ID map, annotated videos and the exact config used.

## Comparing configurations

| Goal | Command |
|---|---|
| Compare pose backends like for like | `python -m posepipe run -c configs/experiments/fair_comparison.yaml -o pose.type=alphapose` (repeat with `rtmpose`) |
| Unsupervised metrics for one run | `python -m posepipe evaluate --root <run_dir>` |
| Plots for one run | `python -m posepipe plots --root <run_dir>` |
| Parameter sweeps | `python -m posepipe study --pose alphapose --variant all --experiments all` |

All options, defaults and output formats are documented in
[`Modular_Pipeline/README.md`](Modular_Pipeline/README.md).

## Repository layout

```
Modular_Pipeline/
  configs/          default.yaml (every option) and experiment presets
  posepipe/         pipeline source: detection, pose, reid, tracking, crosscam, evaluation
  externals/        vendored AlphaPose (pure-PyTorch parts are used; no build step)
  checkpoints/      model weights go here (not tracked)
  tests/            end-to-end tests, run without a GPU or weights
```

## Tests

```bash
cd Modular_Pipeline
python -m unittest discover -s tests -v
```

## Origin

This pipeline merges two earlier prototypes, `AlphaPose_OSNet_Pipeline` and
`RTMO_DeepSORT_Pipeline`. Changes made while merging them, including fixes that affect
results, are listed under "Differences from the two original repositories" in
[`Modular_Pipeline/README.md`](Modular_Pipeline/README.md).

## Third-party code and licences

| Component | Source | Licence |
|---|---|---|
| AlphaPose (vendored in `externals/AlphaPose`) | [MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) | See [`externals/AlphaPose/LICENSE`](Modular_Pipeline/externals/AlphaPose/LICENSE): non-commercial use |
| DeepSORT core (vendored in `posepipe/third_party/deep_sort`) | [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch), based on [nwojke/deep_sort](https://github.com/nwojke/deep_sort) | See [`LICENSE`](Modular_Pipeline/posepipe/third_party/deep_sort/LICENSE) |
| MMPose / RTMPose | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | Apache-2.0 (installed via pip) |
| Ultralytics YOLOv8 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 (installed via pip) |
| torchreid | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | MIT (installed via pip) |

Model weights belong to their respective authors and are downloaded separately.
