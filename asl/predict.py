"""Predict ordered letter images, preserving raw text and uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CLASS_NAMES, LABEL_IDS, load_image


def decode_predictions(probabilities, threshold=0.7, top_k=3):
    if not 0 <= threshold <= 1 or not 1 <= top_k <= len(CLASS_NAMES):
        raise ValueError("threshold must be in [0, 1] and top_k in [1, 24]")
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("Model must return one row of 24 probabilities per image")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("Model returned invalid probabilities")
    results = []
    for row in probabilities:
        ranked = np.argsort(-row)[:top_k]
        best = int(ranked[0])
        accepted = bool(row[best] >= threshold)
        results.append({
            "letter": CLASS_NAMES[best] if accepted else "?",
            "best_guess": CLASS_NAMES[best], "confidence": float(row[best]), "accepted": accepted,
            "alternatives": [{"letter": CLASS_NAMES[int(i)], "confidence": float(row[i])} for i in ranked],
        })
    return results


def predict_images(model_dir, image_paths, threshold=0.7, top_k=3, spellcheck=False):
    import tensorflow as tf

    if not image_paths:
        raise ValueError("Provide at least one image")
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != 1 or metadata.get("class_names") != list(CLASS_NAMES)
            or metadata.get("label_ids") != list(LABEL_IDS)
            or metadata.get("input_shape") != [28, 28, 1]
            or metadata.get("input_range") != [0, 255]
            or metadata.get("normalization") != "inside_model"):
        raise ValueError("Incompatible model metadata. Train with python -m asl.train first.")
    images = np.stack([load_image(path) for path in image_paths])
    model = tf.keras.models.load_model(model_dir / "model.keras", compile=False)
    if tuple(model.input_shape[1:]) != (28, 28, 1) or model.output_shape[-1] != 24:
        raise ValueError("Model shape does not match the saved metadata")
    results = decode_predictions(model.predict(images, verbose=0), threshold, top_k)
    for path, result in zip(image_paths, results):
        result["image"] = str(path)
    text = "".join(result["letter"] for result in results)
    output = {"raw_text": text, "predictions": results}
    if spellcheck:
        if "?" in text:
            output["spelling_suggestion"] = None
            output["spelling_note"] = "Resolve uncertain letters before requesting a spelling suggestion."
        else:
            try:
                from spellchecker import SpellChecker
            except ImportError as error:
                raise ValueError("Install pyspellchecker to use --spellcheck") from error
            output["spelling_suggestion"] = SpellChecker().correction(text.lower()) or text
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="One image per letter, in word order")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/run-1"))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--spellcheck", action="store_true", help="Suggest a spelling separately from raw predictions")
    args = parser.parse_args()
    try:
        result = predict_images(args.model_dir, args.images, args.threshold, args.top_k, args.spellcheck)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
