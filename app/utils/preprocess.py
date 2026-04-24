"""Preprocessing helpers for emotion inference."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_face(face_bgr: np.ndarray, target_size: tuple[int, int] = (48, 48)) -> np.ndarray:
    """Convert face patch to normalized grayscale tensor shape (1, H, W, 1)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[None, ..., None]
