# ASL alphabet recognition

A CNN that recognizes **24 static ASL letter classes** from Sign Language MNIST:
`A B C D E F G H I K L M N O P Q R S T U V W X Y`.
The model does **not** recognize J or Z, whole sentences, continuous signing,
or the `space`, `del`, and `nothing` gestures from the separate ASL Alphabet dataset.

## Features

- A TensorFlow CNN with a pretrained model for 24 letter classes.
- Image prediction with confidence scores, alternative guesses, and a configurable
  threshold for uncertain predictions.
- Letter sequences combined into text, with optional spelling suggestions.
- Pixel normalization inside the model for consistent training and prediction.
- Training with rotation, translation, and zoom augmentation, a stratified
  validation split, early stopping, learning-rate reduction, and checkpoint selection.
- Evaluation reports, confusion matrices, training history, and reproducible split indices.
- Automated tests and continuous integration with GitHub Actions.

## Setup

Use Python **3.11 or 3.12**. From this repository directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux activate with `source .venv/bin/activate` instead.
TensorFlow can train on CPU; GPU availability depends on your environment.
TensorFlow and Keras are pinned to match the included checkpoint. Other dependency
ranges are compatibility constraints, not an exact lockfile.

## Use the trained model

A trained checkpoint and its class/preprocessing metadata are included in
`models/sign-language-mnist/`. Put a cropped hand-sign image at `images/test.jpg`,
then run:

```powershell
python -m asl.predict --model-dir models/sign-language-mnist images/test.jpg
```

The checkpoint uses Python 3.12.2, TensorFlow 2.21.0, and Keras 3.15.1.
Install `requirements.txt` before loading it to use compatible package versions.
The default `--model-dir artifacts/run-1` is for your own training runs; use the
explicit `--model-dir` above for the included checkpoint.

Training is optional if you want to use the included model.

## Model performance

| Metric | Value |
| --- | --- |
| Held-out test accuracy | **99.67%** |
| Test loss | **0.01133** |
| Training images | 23,336 |
| Validation images | 4,119 |
| Test images | 7,172 |
| Training epochs | 20 |
| Selected checkpoint | Epoch 19, lowest validation loss |

The test set is reserved for evaluation after checkpoint selection. Detailed
reports, confusion counts, training history, and split indices are in `results/run-1/`.
The automated test suite checks functionality; the accuracy above comes from
evaluation on Sign Language MNIST.

### External photo examples

Predictions for four manually labeled external photos:

| Expected letter | Output at threshold 0.7 | Highest-scoring class | Score |
| --- | --- | --- | --- |
| B | B | B | 98.17% |
| K | T | T | 99.09% |
| P | P | P | approximately 100% |
| Q | ? (rejected) | K | 56.54% |

These four examples are illustrative, not a representative accuracy estimate.
They show that the model can be confidently wrong on external photos. High
dataset accuracy does not establish reliable performance on arbitrary camera images.
The external photos are not included in the repository.

## Train

Download Sign Language MNIST automatically (requires internet; Kaggle may request authentication):

```powershell
python -m asl.train --download --output-dir artifacts/run-1 --epochs 20
```

Or use CSV files you have already downloaded:

```powershell
python -m asl.train --data-dir "data/sign-language-mnist" --output-dir artifacts/run-1
```

`--data-dir` must contain `sign_mnist_train.csv` and `sign_mnist_test.csv`, directly
or in subfolders. CSVs must contain `label` and `pixel1` through `pixel784`.
The loader searches the supplied directory and its subfolders.

Options include `--epochs`, `--batch-size`, `--validation-fraction` (default 0.15),
and `--seed` (default 42). The validation fraction applies only to the training
CSV. Use a **new output directory for each run**; existing runs are not overwritten.
The fixed seed aids repeatability, but exact floating-point results can vary by hardware.

Artifacts:

| File | Meaning |
| --- | --- |
| `model.keras` | Full model from the epoch with lowest validation loss |
| `metadata.json` | Class order, preprocessing, settings, sample counts, and best epoch |
| `metrics.json` | Held-out test loss and accuracy for that checkpoint |
| `classification_report.json` | Per-letter precision, recall, and F1 |
| `confusion_matrix.csv` | Rows = actual, columns = predicted, in the documented class order |
| `history.json`, `training.csv` | Learning curves as numbers |
| `split_indices.npz` | Training and validation row indices in the source training CSV |

Keep `model.keras` and `metadata.json` together for prediction. An interrupted
run may contain partial artifacts; it is ready for prediction only after metadata is written.

## Predict letters or a word

Supply **one cropped hand image per letter**, in word order:

```powershell
python -m asl.predict --model-dir models/sign-language-mnist images/H.png images/E.png images/L.png images/L.png images/O.png
```

Optional spelling suggestion:

```powershell
python -m asl.predict --model-dir models/sign-language-mnist --spellcheck --threshold 0.7 --top-k 3 images/A.png
```

Output is JSON with `raw_text`, individual predictions, the highest-scoring guess,
confidence, and alternatives. Below the threshold, `raw_text` includes `?`;
spell correction is skipped when any letter is uncertain.
Scores are model softmax values, **not calibrated correctness probabilities**.
The default threshold is a starting point; tune it using representative validation images.

`python Model.py ...` and `python Prediction.py ...` also work as convenience entry points.

## Project structure

```text
asl/
  data.py                    CSV validation, label mapping, and image loading
  network.py                 CNN architecture and embedded preprocessing
  train.py                   Training and evaluation
  predict.py                 Image prediction and spelling suggestions
models/sign-language-mnist/  Pretrained checkpoint and model metadata
results/run-1/               Evaluation reports and training history
tests/                      Automated tests
legacy/                     Archived scripts
Model.py                    Training command entry point
Prediction.py               Prediction command entry point
requirements.txt            Python dependencies
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover class mapping, malformed CSVs, image preprocessing, disjoint stratified
splits, uncertainty handling, and a small synthetic training/save/reload/prediction
run. An additional test loads the included pretrained model for image prediction.
The synthetic run verifies plumbing; its accuracy is not a model benchmark.
Tests do not download the dataset.

## Limitations

Sign Language MNIST images differ from arbitrary camera photos. Cropping and
resizing alone do not guarantee useful predictions on new lighting, backgrounds,
or signers. A row-level validation split also does not establish performance on
unseen signers or capture sessions. The system classifies individual static letter
images; it does not translate continuous sign language or recognize motion-based signs.

## References

- [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [Keras Rescaling layer](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/)
- [Keras ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)
