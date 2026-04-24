"""CLI entrypoint for real-time emotion recognition."""
from __future__ import annotations

import argparse

from app.config import Paths, RuntimeConfig
from app.inference.realtime import RealtimeEmotionApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face emotion detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device id")
    parser.add_argument("--min-face-conf", type=float, default=0.6, help="Minimum face confidence [0,1]")
    parser.add_argument("--no-fps", action="store_true", help="Disable FPS overlay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RuntimeConfig(
        webcam_id=args.camera,
        detection_confidence=args.min_face_conf,
        display_fps=not args.no_fps,
    )
    app = RealtimeEmotionApp(Paths(), cfg)
    app.run()


if __name__ == "__main__":
    main()
