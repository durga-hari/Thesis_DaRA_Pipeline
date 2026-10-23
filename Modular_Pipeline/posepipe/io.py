from __future__ import annotations

import json
from pathlib import Path

import cv2


class VideoReader:
    def __init__(self, path, start_frame: int = 0):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame

    def close(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, path, fps: float, size_wh, enabled: bool = True):
        self.enabled = enabled
        self.vw = None
        if enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size_wh))

    def write(self, frame):
        if self.vw is not None:
            self.vw.write(frame)

    def close(self):
        if self.vw is not None:
            self.vw.release()


class JsonlWriter:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, obj):
        self.f.write(json.dumps(obj) + "\n")

    def close(self):
        self.f.close()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
