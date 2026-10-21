# ASL alphabet recognition

A CNN that recognizes **24 static ASL letter classes** from Sign Language MNIST:
`A B C D E F G H I K L M N O P Q R S T U V W X Y`.
The model does **not** recognize J or Z, whole sentences, continuous signing,
or the `space`, `del`, and `nothing` gestures from the separate ASL Alphabet dataset.

## What changed

- Runnable Python modules replace notebook-only commands and machine-specific paths.
- A stratified validation split comes from the training CSV; the supplied test CSV
  is used only after training to evaluate the selected checkpoint.
- Pixel normalization lives inside the saved model, so training and prediction
  both take raw grayscale pixels in `[0, 255]`.
- The original CNN architecture is retained, with training-only rotation,
  translation, and zoom augmentation, checkpointing, early stopping, and learning-rate reduction.
- Saved class metadata gives the exact correspondence between outputs and letters.
- Predictions include confidence scores, alternatives, and `?` for uncertain letters.
  Spell correction is optional and never replaces the raw prediction.
- Evaluation reports, confusion matrix counts, training history, split indices,
  regression tests, and a GitHub Actions workflow make results easier to inspect.

The original repository scripts are preserved unchanged under `legacy/`.
Existing `.h5` models do not use the new preprocessing contract. Use the included
checkpoint below, or retrain with this version.

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

The checkpoint was trained using Python 3.12.2, TensorFlow 2.21.0, and Keras 3.15.1.
Install `requirements.txt` before loading it; older Keras versions may not read
this checkpoint's serialization format.
The default `--model-dir artifacts/run-1` is for your own training runs; use the
explicit `--model-dir` above for the included checkpoint.

Training is optional if you want to use the included model.

## Recorded results

The included model achieved **99.6654% accuracy (99.67% rounded)** and
**0.01133 loss** on 7,172 held-out Sign Language MNIST test images.
The supplied training CSV was split into 23,336 training and 4,119 validation
images. Training ran for 20 epochs; the checkpoint from epoch 19 had the lowest
validation loss. See `results/run-1/` for evaluation reports, confusion counts,
history, and split indices. These are recorded local results, not a fresh CI benchmark.

Four manually labeled external photos were also tried:

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
No absolute Colab/cache path is assumed.

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
python -m asl.predict --model-dir artifacts/run-1 images/H.png images/E.png images/L.png images/L.png images/O.png
```

Optional spelling suggestion:

```powershell
python -m asl.predict --model-dir artifacts/run-1 --spellcheck --threshold 0.7 --top-k 3 images/A.png
```

Output is JSON with `raw_text`, individual predictions, the highest-scoring guess,
confidence, and alternatives. Below the threshold, `raw_text` includes `?`;
spell correction is skipped when any letter is uncertain.
Scores are model softmax values, **not calibrated correctness probabilities**.
The default threshold is a starting point; tune it using representative validation images.

`python Model.py ...` and `python Prediction.py ...` also work as convenience entry points.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover class mapping, malformed CSVs, image preprocessing, disjoint stratified
splits, uncertainty handling, and a small synthetic training/save/reload/prediction
run. An additional test loads the included pretrained model for image prediction.
The synthetic run verifies plumbing; its accuracy is not a model benchmark.
Tests do not download the dataset.

## Evaluation limits and next steps

The old approximately 99.8% result used the test set during training for validation
and learning-rate decisions. It is not an independent benchmark for this revision.
The new held-out result is recorded above; it is not a direct accuracy improvement
comparison with the old evaluation protocol.
Choose hyperparameters with the validation set, rather than repeatedly consulting the test set.

Sign Language MNIST images differ from arbitrary camera photos. Cropping and
resizing alone do not guarantee useful predictions on new lighting, backgrounds,
or signers. A row-level validation split also does not establish performance on
unseen signers or capture sessions. Do not interpret this as a sign-language translator.

Future work: evaluate a separate real-camera dataset; add hand localization and
background handling; then consider a live camera interface. Supporting J/Z and
continuous signing requires suitable temporal data and a different evaluation setup.

## References

- [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [Keras Rescaling layer](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/)
- [Keras ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)
