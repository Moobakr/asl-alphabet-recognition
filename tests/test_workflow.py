"""Regression tests, including a real TensorFlow save/reload smoke test."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from asl.data import CLASS_NAMES, LABEL_IDS, PIXEL_COLUMNS, find_csv, load_csv, load_image, split_training
from asl.predict import decode_predictions, predict_images


def sample_frame(samples_per_class=8):
    rng = np.random.default_rng(42)
    labels = np.repeat(LABEL_IDS, samples_per_class)
    frame = pd.DataFrame(rng.integers(0, 256, (len(labels), 784)), columns=PIXEL_COLUMNS)
    frame.insert(0, "label", labels)
    return frame


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_dataset_label_mapping_skips_j_and_z(self):
        path = self.root / "train.csv"
        frame = sample_frame(1)
        frame["pixel1"] = 11
        frame["pixel784"] = 99
        frame[frame.columns[::-1]].to_csv(path, index=False)
        images, targets = load_csv(path)
        self.assertEqual(CLASS_NAMES, tuple("ABCDEFGHIKLMNOPQRSTUVWXY"))
        np.testing.assert_array_equal(targets, np.arange(24))
        self.assertEqual(images.shape, (24, 28, 28, 1))
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(images[0, 0, 0, 0], 11)
        self.assertEqual(images[0, -1, -1, 0], 99)

    def test_bad_labels_pixels_and_missing_columns_are_rejected(self):
        for problem in ("label", "fractional_label", "negative", "large", "nan", "columns", "empty"):
            with self.subTest(problem=problem):
                frame = sample_frame(1)
                if problem == "label":
                    frame.loc[0, "label"] = 9
                elif problem == "fractional_label":
                    frame["label"] = frame["label"].astype(float)
                    frame.loc[0, "label"] = 0.5
                elif problem in ("negative", "large", "nan"):
                    frame["pixel1"] = frame["pixel1"].astype(float)
                    frame.loc[0, "pixel1"] = {"negative": -1, "large": 256, "nan": np.nan}[problem]
                elif problem == "columns":
                    frame = frame.drop(columns="pixel1")
                else:
                    frame = frame.iloc[:0]
                path = self.root / "bad.csv"
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_csv(path)

    def test_validation_is_disjoint_stratified_and_repeatable(self):
        targets = np.repeat(np.arange(24), 20)
        images = np.zeros((len(targets), 28, 28, 1))
        training, validation = split_training(images, targets, 0.2, 7)
        training2, validation2 = split_training(images, targets, 0.2, 7)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(training) | set(validation), set(range(len(targets))))
        self.assertEqual(set(targets[training]), set(range(24)))
        np.testing.assert_array_equal(np.bincount(targets[validation]), np.full(24, 4))
        np.testing.assert_array_equal(training, training2)
        np.testing.assert_array_equal(validation, validation2)

    def test_missing_classes_and_invalid_fraction_are_rejected(self):
        images = np.zeros((100, 28, 28, 1))
        with self.assertRaises(ValueError):
            split_training(images, np.zeros(100))
        with self.assertRaises(ValueError):
            split_training(images, np.zeros(100), 0)
        with self.assertRaises(ValueError):
            split_training(images, np.zeros(99))

    def test_image_and_csv_use_same_pixel_scale(self):
        path = self.root / "white.png"
        Image.new("RGB", (70, 50), "white").save(path)
        image = load_image(path)
        self.assertEqual(image.shape, (28, 28, 1))
        self.assertTrue((image == 255).all())
        with self.assertRaises(FileNotFoundError):
            load_image(self.root / "missing.png")

    def test_nested_csv_discovery_and_missing_file(self):
        nested = self.root / "nested"
        nested.mkdir()
        path = nested / "sign_mnist_test.csv"
        path.touch()
        self.assertEqual(find_csv(self.root, path.name), path)
        with self.assertRaises(FileNotFoundError):
            find_csv(self.root, "missing.csv")


class PredictionTests(unittest.TestCase):
    def test_uncertainty_and_label_order(self):
        scores = np.full((2, 24), 0.2 / 23)
        scores[0, 9] = 0.8
        scores[1] = 1 / 24
        decoded = decode_predictions(scores)
        self.assertEqual(decoded[0]["letter"], "K")
        self.assertTrue(decoded[0]["accepted"])
        self.assertEqual(decoded[1]["letter"], "?")
        self.assertEqual(len(decoded[0]["alternatives"]), 3)

    def test_invalid_scores_and_options(self):
        valid = np.full((1, 24), 1 / 24)
        for scores in (np.zeros((1, 25)), np.zeros((1, 24)), np.full((1, 24), np.nan),
                       np.full((1, 24), -1)):
            with self.assertRaises(ValueError):
                decode_predictions(scores)
        for options in ({"threshold": -0.1}, {"threshold": 1.1}, {"top_k": 0}, {"top_k": 25}):
            with self.assertRaises(ValueError):
                decode_predictions(valid, **options)


class IntegrationTests(unittest.TestCase):
    def test_packaged_checkpoint_predicts_with_saved_metadata(self):
        model_dir = Path(__file__).resolve().parents[1] / "models" / "sign-language-mnist"
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.png"
            Image.new("L", (40, 40), 128).save(image_path)
            result = predict_images(model_dir, [image_path], threshold=0)
        self.assertEqual(len(result["raw_text"]), 1)
        self.assertIn(result["raw_text"], CLASS_NAMES)
        self.assertTrue(result["predictions"][0]["accepted"])

    def test_train_save_reload_and_predict(self):
        import tensorflow as tf
        from asl.train import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_frame(8).to_csv(root / "sign_mnist_train.csv", index=False)
            sample_frame(1).to_csv(root / "sign_mnist_test.csv", index=False)
            output = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = train(root, output, epochs=1, batch_size=32, validation_fraction=0.25)
            self.assertTrue(np.isfinite(metrics["loss"]))
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["train_samples"], 144)
            self.assertEqual(metadata["validation_samples"], 48)
            self.assertEqual(metadata["test_samples"], 24)
            with np.load(output / "split_indices.npz") as split:
                self.assertFalse(set(split["train"]) & set(split["validation"]))
            saved = tf.keras.models.load_model(output / "model.keras", compile=False)
            np.testing.assert_allclose(saved.get_layer("normalize_pixels")(np.full((1, 28, 28, 1), 255)), 1)
            batch = np.random.default_rng(7).uniform(0, 255, (2, 28, 28, 1)).astype(np.float32)
            # Augmentation and dropout must be disabled for inference.
            np.testing.assert_allclose(saved(batch, training=False), saved(batch, training=False))
            image_path = root / "letter.png"
            Image.new("L", (28, 28), 128).save(image_path)
            result = predict_images(output, [image_path], threshold=1, spellcheck=True)
            self.assertEqual(result["raw_text"], "?")
            self.assertIsNone(result["spelling_suggestion"])
            self.assertEqual(len(result["predictions"]), 1)
            self.assertEqual(int(np.loadtxt(output / "confusion_matrix.csv", delimiter=",", skiprows=1).sum()), 24)
            with self.assertRaises(ValueError):
                train(root, output, epochs=1)
            metadata["normalization"] = "outside_model"
            (output / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                predict_images(output, [image_path])


if __name__ == "__main__":
    unittest.main()
