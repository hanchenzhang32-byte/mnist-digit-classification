"""MNIST CNN: train, validate, test and visualise a small convolutional network.

Uses the same data split, preprocessing, optimiser, seed and evaluation code as
train.py (the MLP), so the two models can be compared directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

# Shared helpers from the MLP script: device choice, preprocessing,
# training/evaluation loops and figure generation.
from train import as_tensors, choose_device, evaluate, save_figures, train_epoch

ROOT = Path(__file__).resolve().parent


class CNN(nn.Module):
    """Two conv blocks (32 and 64 filters) followed by a small classifier head."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28×32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14×14×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14×14×64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 7×7×64
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_cnn")
    parser.add_argument("--eval-only", action="store_true",
                        help="load best_model.pt from output-dir and evaluate on the test set")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        parser.error("epochs, batch-size, lr and threads must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.output_dir / "train.log", encoding="utf-8")])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    logging.info("PyTorch %s | device=%s | seed=%d | model=CNN", torch.__version__, device, args.seed)

    raw_train = MNIST(root=args.data_dir, train=True, download=True)
    raw_test = MNIST(root=args.data_dir, train=False, download=True)
    test_loader = DataLoader(as_tensors(raw_test), batch_size=args.batch_size, shuffle=False)
    checkpoint_path = args.output_dir / "best_model.pt"
    model = CNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    history = []
    start = time.perf_counter()

    if not args.eval_only:
        # Identical split to train.py: 55,000 train / 5,000 validation, seeded.
        train_set, val_set = random_split(as_tensors(raw_train), [55000, 5000],
                                          generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_accuracy = -1.0
        logging.info("Split: 55,000 train / 5,000 val / 10,000 test | parameters %d", n_params)
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, val_accuracy, _ = evaluate(model, val_loader, device)
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                       val_loss=val_loss, val_accuracy=val_accuracy,
                       seconds=time.perf_counter() - epoch_start)
            history.append(row)
            logging.info("Epoch %d/%d | train loss=%.4f acc=%.2f%% | val loss=%.4f acc=%.2f%% | %.1fs",
                         epoch, args.epochs, train_loss, train_accuracy * 100,
                         val_loss, val_accuracy * 100, row["seconds"])
            if val_accuracy > best_accuracy:  # checkpoint chosen on validation only
                best_accuracy = val_accuracy
                torch.save({"model_type": "CNN", "model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_accuracy": val_accuracy, "seed": args.seed,
                            "normalization": {"mean": 0.1307, "std": 0.3081},
                            "config": {key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items()}}, checkpoint_path)
            with (args.output_dir / "history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, predictions = evaluate(model, test_loader, device, True)
    save_figures(history, raw_test, predictions, args.output_dir)
    metrics = {"model_type": "CNN",
               "architecture": "Conv(1→32,3×3)-ReLU-MaxPool → Conv(32→64,3×3)-ReLU-MaxPool"
                               " → FC(3136→128)-ReLU-Dropout(0.2) → FC(128→10)",
               "parameters": n_params,
               "device": str(device), "pytorch_version": str(torch.__version__),
               "best_epoch": checkpoint["epoch"], "seed": checkpoint["seed"],
               "best_val_accuracy": checkpoint["val_accuracy"],
               "test_loss": test_loss, "test_accuracy": test_accuracy,
               "test_correct": int((predictions == raw_test.targets).sum()),
               "train_samples": 55000, "val_samples": 5000, "test_samples": len(raw_test),
               "elapsed_seconds": time.perf_counter() - start}
    metrics_path = args.output_dir / ("evaluation.json" if args.eval_only else "metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Best epoch=%d | test loss=%.4f | test accuracy=%.2f%% (%d/%d)",
                 checkpoint["epoch"], test_loss, test_accuracy * 100,
                 metrics["test_correct"], len(raw_test))


if __name__ == "__main__":
    main()
