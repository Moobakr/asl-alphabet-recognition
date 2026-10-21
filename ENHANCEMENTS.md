# Enhancement review

Based on repository commit `fa80a58` (`main` at download time).

## Changes

1. Preserved the original extensionless scripts unchanged under `legacy/`.
2. Added portable `Model.py` and `Prediction.py` entry points backed by the `asl` package.
3. Kept the original 24-class CNN; moved normalization and augmentation into the model.
4. Separated validation from the supplied test CSV and evaluated the best saved checkpoint.
5. Added metadata, per-class evaluation, confusion counts, training history, and split indices.
6. Added confidence thresholds and alternative guesses; kept spelling suggestions separate.
7. Added dependency declarations, setup instructions, regression tests, and CI configuration.
8. Included the trained checkpoint with portable metadata and recorded evaluation artifacts.

## Local validation

- Initial development checks used Python 3.12.14 and TensorFlow 2.18.0, CPU.
- Publication checks use Python 3.12.2, TensorFlow 2.21.0, and Keras 3.15.1.
- Ten automated tests cover one actual training epoch on synthetic data,
  checkpoint reload, packaged-model inference, metadata validation, and uncertainty handling.
- Both command-line help screens and Python compilation checked.
- Original archived scripts have no content changes.

## Trained checkpoint

The user trained the revised pipeline on Sign Language MNIST with Python 3.12.2
and TensorFlow 2.21.0 / Keras 3.15.1: 23,336 training, 4,119 validation, and 7,172 test samples.
Training completed 20 epochs; epoch 19 was selected by validation loss.
Recorded test accuracy is 0.99665367603302 and loss is 0.01133091188967228.

The model and portable metadata are included in `models/sign-language-mnist/`.
Evaluation artifacts are in `results/run-1/`. The README also records four
external-photo checks, including a confident incorrect prediction and an uncertain
rejection. These examples are not a representative benchmark.

Dataset files, the virtual environment, and the external photos are not bundled.
