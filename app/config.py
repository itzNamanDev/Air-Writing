"""Application configuration for real-time emotion detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


@dataclass(frozen=True)
class Paths:
    """Filesystem paths used throughout the project."""

    root: Path = Path(__file__).resolve().parents[1]
    assets_dir: Path = root / "assets"
    logs_dir: Path = root / "logs"
    captures_dir: Path = root / "captures"

    face_proto: Path = assets_dir / "deploy.prototxt"
    face_model: Path = assets_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model_weights: Path = assets_dir / "emotion_cnn.keras"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime parameters for webcam and model inference."""

    input_size: tuple[int, int] = (48, 48)
    detection_confidence: float = 0.60
    webcam_id: int = 0
    display_fps: bool = True
    save_on_emotion: tuple[str, ...] = ("Surprise",)
    screenshot_cooldown_seconds: float = 2.0
