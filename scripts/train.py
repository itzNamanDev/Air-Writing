"""Train emotion CNN on FER2013 and save best weights."""
from __future__ import annotations

import argparse

from app.config import Paths
from app.data.fer2013 import DatasetConfig, load_fer2013
from app.models.emotion_cnn import build_emotion_cnn, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train emotion CNN on FER2013")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths()

    train_ds, val_ds, test_ds = load_fer2013(DatasetConfig(batch_size=args.batch_size))
    model = build_emotion_cnn(input_shape=(48, 48, 1), num_classes=7)

    train_model(model, train_ds, val_ds, paths.model_weights, epochs=args.epochs)

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
