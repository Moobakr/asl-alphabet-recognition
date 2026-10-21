"""The original CNN, with preprocessing saved inside the model."""

import tensorflow as tf

from .data import CLASS_NAMES


def build_model(seed=42):
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Rescaling(1.0 / 255, name="normalize_pixels"),
        layers.RandomRotation(10 / 360, seed=seed),
        layers.RandomTranslation(0.1, 0.1, seed=seed + 1),
        layers.RandomZoom(0.1, seed=seed + 2),
        layers.Conv2D(75, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, padding="same"),
        layers.Conv2D(50, 3, padding="same", activation="relu"),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, padding="same"),
        layers.Conv2D(25, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, padding="same"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="asl_mnist_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
