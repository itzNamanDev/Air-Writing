"""CSV emotion event logger."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


class EmotionLogger:
    """Append emotion detections to a CSV file for later analysis."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc", "emotion", "confidence", "face_index"])

    def log(self, emotion: str, confidence: float, face_index: int) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, emotion, f"{confidence:.5f}", face_index])
