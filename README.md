# MNIST Handwritten Digit Classification with PyTorch

A complete training and evaluation workflow for a multilayer perceptron (MLP) that recognises handwritten digits. The project includes dataset preparation, validation-based checkpoint selection, independent test evaluation and visual diagnostics.

## Recorded results

| Setting | Recorded value |
| --- | --- |
| Architecture | 784 → 256 → 128 → 10, ReLU, Dropout (0.2) |
| Training / validation / test images | 55,000 / 5,000 / 10,000 |
| Optimiser | Adam, learning rate 0.001 |
| Epochs / batch size / seed | 5 / 128 / 42 |
| Best validation accuracy | 97.50% |
| Independent test accuracy | **97.82% (9,782 / 10,000)** |
| Test cross-entropy loss | 0.06896 |
| Parameters | 235,146 |

The checkpoint was chosen using validation accuracy. Test data was held out from training and checkpoint selection. The saved model was independently re-evaluated on CPU; see [metrics.json](metrics.json) and [evaluation.json](evaluation.json).

These results describe a single recorded run on MNIST, a controlled benchmark. Hardware and library differences may affect results. This repository contains the completed MLP implementation; earlier CNN experiments are separate and are not included here.

## Visual results

![Training and validation curves](training_curves.png)

![Test confusion matrix](confusion_matrix.png)

![Sample test predictions](predictions.png)

## Run locally

Python 3.14 was used for the recorded run. The direct dependency versions are in `requirements.txt`; `requirements-lock.txt` records the original environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

The script downloads MNIST via Torchvision and automatically selects CUDA, Apple MPS or CPU. To select CPU explicitly:

```bash
python train.py --device cpu
```

To download the data without training:

```bash
python train.py --download-only
```

After training, re-evaluate the saved checkpoint without retraining:

```bash
python train.py --eval-only --device cpu
```

Use a separate output directory to preserve an existing run:

```bash
python train.py --epochs 10 --output-dir outputs_10epochs
```

## Repository contents

- `train.py`: data preparation, model definition, training, evaluation and visualisation.
- `requirements.txt` / `requirements-lock.txt`: dependency versions.
- `metrics.json` / `evaluation.json`: recorded training-run and independent reload results.
- `history.csv`: per-epoch training and validation measurements.
- `training_curves.png`, `confusion_matrix.png`, `confusion_matrix.csv`, `predictions.png`: recorded evaluation artefacts.

Fresh runs write to `outputs/`; downloaded datasets, checkpoints, logs and local environment files are excluded by `.gitignore`. The result files at the repository root are the recorded run snapshot.

## Methods

Images are normalised using mean 0.1307 and standard deviation 0.3081. The network outputs logits for cross-entropy loss. Dropout is disabled for evaluation. The training-data split and shuffled loader use seeded generators.

## References

- [Torchvision MNIST dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [PyTorch](https://pytorch.org/)
