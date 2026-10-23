"""
Top-down pose estimators.

Interface (every backend):

    est = build_pose(cfg)
    results = est(frame_bgr, boxes_xyxy)   # boxes: (N, 4) float
    # results[i] corresponds to boxes[i]:
    #   {"keypoints": np.ndarray (J, 3) [x, y, conf] in image coords,
    #    "score": float}                    # mean joint confidence
    est.num_joints  -> 17 (COCO) or 133 (COCO-WholeBody)

Frames are BGR (OpenCV order) everywhere; each backend converts internally.
"""

from ..registry import Registry

POSE = Registry("pose")
POSE.register_lazy("alphapose", "posepipe.pose.alphapose_hrnet:AlphaPoseEstimator",
                   aliases=("alphapose_hrnet", "hrnet"))
POSE.register_lazy("rtmpose", "posepipe.pose.rtmpose:RTMPoseEstimator",
                   aliases=("rtmo", "mmpose"))  # 'rtmo' kept so old configs still load


def build_pose(cfg: dict):
    return POSE.build(cfg)
