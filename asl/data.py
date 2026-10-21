"""Dataset validation and the shared input/label contract."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

LABEL_IDS = tuple(i for i in range(25) if i != 9)
CLASS_NAMES = tuple(chr(65 + i) for i in LABEL_IDS)
PIXEL_COLUMNS = [f"pixel{i}" for i in range(1, 785)]


def find_csv(root, filename):
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    direct = root / filename
    matches = sorted(root.rglob(filename), key=lambda p: (len(p.parts), str(p)))
    if direct.is_file():
        return direct
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {root}")
    return matches[0]


def load_csv(path):
    frame = pd.read_csv(path)
    expected = {"label", *PIXEL_COLUMNS}
    if set(frame.columns) != expected:
        raise ValueError(f"{path}: expected label and pixel1 through pixel784 columns")
    if frame.empty:
        raise ValueError(f"{path}: dataset is empty")
    raw_labels = frame["label"].to_numpy(dtype=np.float64)
    if not np.isin(raw_labels, LABEL_IDS).all():
        raise ValueError(f"{path}: labels must be 0..24 excluding 9 (J)")
    pixels = frame[PIXEL_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 255).any():
        raise ValueError(f"{path}: pixels must be finite values in [0, 255]")
    # Keep raw pixel values: the model itself owns normalization.
    images = pixels.reshape(-1, 28, 28, 1)
    targets = np.searchsorted(LABEL_IDS, raw_labels).astype(np.int32)
    return images, targets


def split_training(images, targets, validation_fraction=0.15, seed=42):
    if len(images) != len(targets):
        raise ValueError("Each training image must have exactly one target")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if set(np.unique(targets)) != set(range(len(CLASS_NAMES))):
        raise ValueError("Training CSV must contain all 24 supported classes")
    indices = np.arange(len(targets))
    train_indices, validation_indices = train_test_split(
        indices, test_size=validation_fraction, random_state=seed, stratify=targets
    )
    return train_indices, validation_indices


def load_image(path):
    """Match the model's raw 28x28 grayscale input; do not divide by 255."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((28, 28), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32).reshape(28, 28, 1)
