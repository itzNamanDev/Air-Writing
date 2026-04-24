"""OpenCV DNN face detector wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class DnnFaceDetector:
    """Fast DNN-based face detector using OpenCV's SSD Caffe model."""

    def __init__(self, prototxt: Path, model: Path, conf_threshold: float = 0.6) -> None:
        if not prototxt.exists() or not model.exists():
            raise FileNotFoundError(
                "Face detector files are missing. Run `python scripts/download_assets.py` first."
            )

        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        """Return detected face boxes for a BGR frame."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results: list[FaceBox] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.conf_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(FaceBox(x1, y1, x2, y2, confidence))

        return results
