"""Train with a validation split; evaluate the best checkpoint on the test CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .data import CLASS_NAMES, LABEL_IDS, find_csv, load_csv, split_training


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(data_dir, output_dir, epochs=20, batch_size=128, validation_fraction=0.15, seed=42):
    import tensorflow as tf
    from .network import build_model

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new run directory.")
    train_path = find_csv(data_dir, "sign_mnist_train.csv")
    test_path = find_csv(data_dir, "sign_mnist_test.csv")
    images, targets = load_csv(train_path)
    train_indices, val_indices = split_training(images, targets, validation_fraction, seed)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(seed)
    model.summary()
    # Explicit datasets retain partial batches and avoid generator exhaustion.
    training = tf.data.Dataset.from_tensor_slices((images[train_indices], targets[train_indices]))
    training = training.shuffle(len(train_indices), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation = tf.data.Dataset.from_tensor_slices((images[val_indices], targets[val_indices]))
    validation = validation.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    checkpoint = output / "model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(str(output / "training.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(training, validation_data=validation, epochs=epochs, callbacks=callbacks, shuffle=False)
    if not checkpoint.exists() or not all(np.isfinite(history.history["val_loss"])):
        raise RuntimeError("Training did not produce a valid checkpoint")
    # Evaluate exactly the checkpoint that prediction will load.
    model = tf.keras.models.load_model(checkpoint)
    test_images, test_targets = load_csv(test_path)
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_targets)).batch(batch_size)
    metrics = model.evaluate(test_data, return_dict=True, verbose=0)
    probabilities = model.predict(test_data, verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(test_targets, predicted, labels=list(range(24)),
                                   target_names=list(CLASS_NAMES), output_dict=True, zero_division=0)
    matrix = confusion_matrix(test_targets, predicted, labels=list(range(24)))
    write_json(output / "metadata.json", {
        "schema_version": 1, "dataset": "datamunge/sign-language-mnist",
        "class_names": list(CLASS_NAMES), "label_ids": list(LABEL_IDS),
        "input_shape": [28, 28, 1], "input_range": [0, 255],
        "normalization": "inside_model", "seed": seed,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "train_csv": str(train_path.resolve()), "test_csv": str(test_path.resolve()),
        "train_samples": len(train_indices), "validation_samples": len(val_indices),
        "test_samples": len(test_targets), "epochs_requested": epochs,
        "epochs_completed": len(history.epoch), "batch_size": batch_size,
        "validation_fraction": validation_fraction, "checkpoint_monitor": "val_loss",
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
    })
    write_json(output / "metrics.json", {k: float(v) for k, v in metrics.items()})
    write_json(output / "classification_report.json", report)
    write_json(output / "history.json", {k: [float(v) for v in values] for k, values in history.history.items()})
    np.savetxt(output / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d",
               header=",".join(CLASS_NAMES), comments="")
    np.savez_compressed(output / "split_indices.npz", train=train_indices, validation=val_indices)
    print(f"Held-out test accuracy: {metrics['accuracy']:.2%}")
    print(f"Model and evaluation artifacts saved to {output.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing the two Sign Language MNIST CSVs")
    parser.add_argument("--download", action="store_true", help="Download the dataset with kagglehub")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run-1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.data_dir) == args.download:
        parser.error("Choose exactly one of --data-dir or --download")
    try:
        if args.download:
            import kagglehub
            args.data_dir = Path(kagglehub.dataset_download("datamunge/sign-language-mnist"))
        train(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.validation_fraction, args.seed)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
