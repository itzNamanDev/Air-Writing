"""Download required model assets for face detection and optional emotion weights."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from app.config import Paths


FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel",
}


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return
    print(f"[download] {out_path.name}")
    urllib.request.urlretrieve(url, out_path)  # noqa: S310


def main() -> None:
    paths = Paths()
    for filename, url in FILES.items():
        download_file(url, paths.assets_dir / filename)

    print("Done. Face detector assets are ready.")
    print(
        "Note: train emotion weights with `python scripts/train.py` or place your .keras model at "
        f"{paths.model_weights}."
    )


if __name__ == "__main__":
    main()
