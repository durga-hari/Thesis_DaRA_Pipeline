# Checkpoints

Weights are not tracked by git. Paths below match `configs/default.yaml`;
if a file has a different name, change the path in the config instead of renaming.

| File | Used by | Status / source |
|---|---|---|
| `yolov8m.pt` | `detector.yolov8` (default) | Ultralytics release |
| `yolov8s.pt` | lighter detector alternative | Ultralytics release |
| `alphapose/hrnet_w32_256x192.pth` | `pose.alphapose` | **missing**: AlphaPose model zoo (`externals/AlphaPose/docs/MODEL_ZOO.md`, HRNet-W32 row): `gdown 1i63BPlOnp2vSjIZ7ni4Yp3RCPQwqe922 -O checkpoints/alphapose/hrnet_w32_256x192.pth` |
| `rtmpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py` + `.pth` | `pose.rtmpose` | **missing**: `mim download mmpose --config rtmpose-l_8xb32-270e_coco-wholebody-384x288 --dest checkpoints/rtmpose` |
| `reid/resnet50_market_xent.pth.tar` | `reid.resnet50` | **missing**: torchreid model zoo (ResNet-50, Market-1501, softmax) |
| `reid/osnet_x1_0_msmt17.pth` | `reid.osnet` | **missing**: torchreid model zoo (osnet_x1_0, MSMT17) |
| `osnet_x1_0_imagenet.pt` | – | ImageNet-pretrained, **not** ReID-trained; use the MSMT17 weights for real runs |

torchreid model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
