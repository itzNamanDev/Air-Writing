"""CNN model definition, training helpers, and loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import tensorflow as tf


def build_emotion_cnn(input_shape: tuple[int, int, int] = (48, 48, 1), num_classes: int = 7) -> tf.keras.Model:
    """Build a compact CNN optimized for FER-like grayscale emotion inputs."""
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.50)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Create robust training callbacks for stable convergence and checkpointing."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ]


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: Path,
    epochs: int = 40,
) -> tf.keras.callbacks.History:
    """Train the CNN on FER dataset with callbacks and save best checkpoint."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = get_callbacks(model_path)
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)


def load_or_build(model_path: Path, input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Load saved model if available, otherwise return a fresh CNN model."""
    if model_path.exists():
        return tf.keras.models.load_model(model_path)
    return build_emotion_cnn(input_shape=input_shape, num_classes=num_classes)


def predict_batch(model: tf.keras.Model, batch: tf.Tensor) -> tf.Tensor:
    """Run model inference with training disabled."""
    return model(batch, training=False)
