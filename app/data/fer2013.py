"""FER2013 data loading and preprocessing using tensorflow-datasets."""
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
import tensorflow_datasets as tfds


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class DatasetConfig:
    batch_size: int = 64
    image_size: tuple[int, int] = (48, 48)


def _preprocess_sample(sample: dict[str, tf.Tensor], image_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(sample["image"], image_size)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(sample["label"], tf.int32)
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 1.0), label


def load_fer2013(config: DatasetConfig) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train/val/test datasets from TFDS FER2013 splits."""
    train_raw, val_raw, test_raw = tfds.load(
        "fer2013",
        split=["train", "validation", "test"],
        as_supervised=False,
    )

    train_ds = (
        train_raw.map(lambda s: _preprocess_sample(s, config.image_size), num_parallel_calls=AUTOTUNE)
        .map(_augment, num_parallel_calls=AUTOTUNE)
        .shuffle(5000)
        .batch(config.batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_raw.map(lambda s: _preprocess_sample(s, config.image_size), num_parallel_calls=AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_raw.map(lambda s: _preprocess_sample(s, config.image_size), num_parallel_calls=AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds
