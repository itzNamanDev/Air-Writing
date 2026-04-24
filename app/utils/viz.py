"""Visualization utilities for on-screen overlays."""
from __future__ import annotations

import cv2

from app.detection.face_detector import FaceBox


def draw_face_prediction(
    frame,
    face: FaceBox,
    label: str,
    confidence: float,
    color=(0, 220, 0),
):
    """Draw face box and label for one prediction."""
    cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)
    text = f"{label}: {confidence * 100:.1f}%"
    text_origin = (face.x1, max(20, face.y1 - 10))
    cv2.putText(frame, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
    """Draw FPS indicator."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 210, 0),
        2,
        cv2.LINE_AA,
    )
