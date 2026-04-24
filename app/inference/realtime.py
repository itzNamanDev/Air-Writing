"""Real-time webcam inference loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from app.config import EMOTIONS, Paths, RuntimeConfig
from app.detection.face_detector import DnnFaceDetector
from app.models.emotion_cnn import load_or_build
from app.utils.logger import EmotionLogger
from app.utils.preprocess import preprocess_face
from app.utils.viz import draw_face_prediction, draw_fps


@dataclass
class EmotionPrediction:
    label: str
    confidence: float


class RealtimeEmotionApp:
    """End-to-end app for face detection + emotion classification."""

    def __init__(self, paths: Paths, cfg: RuntimeConfig) -> None:
        self.paths = paths
        self.cfg = cfg

        self.detector = DnnFaceDetector(paths.face_proto, paths.face_model, cfg.detection_confidence)
        self.model = load_or_build(paths.model_weights, input_shape=(48, 48, 1), num_classes=len(EMOTIONS))
        self.logger = EmotionLogger(paths.logs_dir / "emotion_log.csv")

        self.last_capture_time = 0.0

    def _predict(self, face_crop: np.ndarray) -> EmotionPrediction:
        batch = preprocess_face(face_crop, self.cfg.input_size)
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        return EmotionPrediction(label=EMOTIONS[idx], confidence=float(probs[idx]))

    def _maybe_save_capture(self, frame: np.ndarray, emotion: str) -> None:
        if emotion not in self.cfg.save_on_emotion:
            return

        now = time.monotonic()
        if now - self.last_capture_time < self.cfg.screenshot_cooldown_seconds:
            return

        self.paths.captures_dir.mkdir(parents=True, exist_ok=True)
        filename = self.paths.captures_dir / f"{emotion.lower()}_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        self.last_capture_time = now

    def run(self) -> None:
        cap = cv2.VideoCapture(self.cfg.webcam_id)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam. Check permissions and camera availability.")

        prev_t = time.perf_counter()
        paused = False

        print("Controls: [q] quit, [p] pause/resume")

        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("Warning: failed to read frame from webcam.")
                    continue

                faces = self.detector.detect(frame)
                for i, face in enumerate(faces):
                    roi = frame[face.y1:face.y2, face.x1:face.x2]
                    if roi.size == 0:
                        continue
                    pred = self._predict(roi)
                    draw_face_prediction(frame, face, pred.label, pred.confidence)
                    self.logger.log(pred.label, pred.confidence, i)
                    self._maybe_save_capture(frame, pred.label)

                if self.cfg.display_fps:
                    now = time.perf_counter()
                    fps = 1.0 / max(now - prev_t, 1e-6)
                    prev_t = now
                    draw_fps(frame, fps)

                if not faces:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (12, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Real-Time Face Emotion Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()
