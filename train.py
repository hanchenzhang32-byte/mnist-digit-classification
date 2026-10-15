"""MNIST MLP：下载、训练、验证、测试以及结果可视化。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import random
import time
import certifi

ROOT = Path(__file__).resolve().parent
# 将绘图库缓存放在项目内，适合受限环境及无桌面的运行环境。
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
# macOS 上的独立 Python 安装可能缺少根证书；保留用户显式设置。
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST


class MLP(nn.Module):
    """展平 28×28 图像，通过全连接层输出 10 类的 logits。"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


def choose_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用，请改用 --device cpu。")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请改用 --device cpu。")
    return torch.device(name)


def as_tensors(dataset):
    # 一次性转为 float32，加速后续取样；MNIST 为单通道灰度图。
    images = dataset.data.unsqueeze(1).float().div_(255)
    images.sub_(0.1307).div_(0.3081)
    return TensorDataset(images, dataset.targets)


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    seen = 0
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach() * labels.size(0)
        total_correct += (logits.detach().argmax(1) == labels).sum()
        seen += labels.size(0)
        if step % 100 == 0:
            logging.info("Epoch %d | batch %d/%d | loss %.4f",
                         epoch, step, len(loader), loss.item())
    return total_loss.item() / seen, total_correct.item() / seen


@torch.inference_mode()
def evaluate(model, loader, device, collect_predictions=False):
    model.eval()  # 关闭 dropout
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    seen, predicted = 0, []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        prediction = logits.argmax(1)
        total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum")
        total_correct += (prediction == labels).sum()
        seen += labels.size(0)
        if collect_predictions:
            predicted.append(prediction.cpu())
    return (total_loss.item() / seen, total_correct.item() / seen,
            torch.cat(predicted) if collect_predictions else None)


def save_figures(history, raw_test, predictions, output_dir):
    if history:
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for split in ("train", "val"):
            axes[0].plot(epochs, [row[f"{split}_loss"] for row in history],
                         marker="o", label=split)
            axes[1].plot(epochs, [row[f"{split}_accuracy"] * 100 for row in history],
                         marker="o", label=split)
        axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
        axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
        for ax in axes:
            ax.set_xticks(epochs)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.savefig(output_dir / "training_curves.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(4, 8, figsize=(12, 7), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        truth, prediction = int(raw_test.targets[i]), int(predictions[i])
        ax.imshow(raw_test.data[i].numpy(), cmap="gray")
        ax.set_title(f"true: {truth} / pred: {prediction}", fontsize=9,
                     color="green" if truth == prediction else "red")
        ax.axis("off")
    fig.savefig(output_dir / "predictions.png", dpi=160)
    plt.close(fig)

    matrix = torch.bincount(raw_test.targets * 10 + predictions, minlength=100).reshape(10, 10)
    with (output_dir / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *range(10)])
        for digit, row in enumerate(matrix.tolist()):
            writer.writerow([digit, *row])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set(title="MNIST test confusion matrix", xlabel="Predicted digit", ylabel="True digit",
           xticks=range(10), yticks=range(10))
    threshold = matrix.max().item() / 2
    for i in range(10):
        for j in range(10):
            count = int(matrix[i, j])
            ax.text(j, i, str(count), ha="center", va="center", fontsize=8,
                    color="white" if count > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="读取 output-dir 中的 best_model.pt 并评估测试集")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        parser.error("epochs、batch-size、lr、threads 必须为正数")
    if args.download_only and args.eval_only:
        parser.error("download-only 和 eval-only 不能同时使用")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.output_dir / "train.log", encoding="utf-8")])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    logging.info("PyTorch %s | device=%s | seed=%d", torch.__version__, device, args.seed)

    # download=True 会检查文件完整性，已下载的文件可离线复用。
    raw_train = MNIST(root=args.data_dir, train=True, download=True)
    raw_test = MNIST(root=args.data_dir, train=False, download=True)
    logging.info("数据已就绪：训练集 %d，测试集 %d | %s",
                 len(raw_train), len(raw_test), args.data_dir.resolve())
    if args.download_only:
        return

    test_loader = DataLoader(as_tensors(raw_test), batch_size=args.batch_size, shuffle=False)
    checkpoint_path = args.output_dir / "best_model.pt"
    model = MLP().to(device)
    history = []
    start = time.perf_counter()

    if not args.eval_only:
        train_set, val_set = random_split(as_tensors(raw_train), [55000, 5000],
                                         generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_accuracy = -1.0
        logging.info("划分：55,000 训练 / 5,000 验证 / 10,000 测试 | 模型参数 %d",
                     sum(p.numel() for p in model.parameters()))
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
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({"model_type": "MLP", "model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_accuracy": val_accuracy, "seed": args.seed,
                            "normalization": {"mean": 0.1307, "std": 0.3081},
                            "config": {key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items()}}, checkpoint_path)
            # 每轮写入，训练中断时仍可查看已完成结果。
            with (args.output_dir / "history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, predictions = evaluate(model, test_loader, device, True)
    save_figures(history, raw_test, predictions, args.output_dir)
    metrics = {"model_type": "MLP", "architecture": [784, 256, 128, 10],
               "device": str(device), "pytorch_version": str(torch.__version__),
               "best_epoch": checkpoint["epoch"], "seed": checkpoint["seed"],
               "best_val_accuracy": checkpoint["val_accuracy"],
               "test_loss": test_loss, "test_accuracy": test_accuracy,
               "test_correct": int((predictions == raw_test.targets).sum()),
               "train_samples": 55000, "val_samples": 5000, "test_samples": len(raw_test),
               "elapsed_seconds": time.perf_counter() - start}
    metrics_path = args.output_dir / ("evaluation.json" if args.eval_only else "metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("最佳 epoch=%d | 测试 loss=%.4f | 测试准确率=%.2f%% (%d/%d)",
                 checkpoint["epoch"], test_loss, test_accuracy * 100,
                 metrics["test_correct"], len(raw_test))
    logging.info("模型：%s | 结果：%s", checkpoint_path.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
