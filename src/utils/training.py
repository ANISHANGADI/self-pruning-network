from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from engine.tensor import Tensor, log_softmax, nll_loss


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
):
    idx = np.arange(x.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = idx[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def accuracy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == targets))


def evaluate_model(model, x: np.ndarray, y: np.ndarray, batch_size: int = 512) -> tuple[float, float]:
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(Tensor(xb, requires_grad=False))
        log_probs = log_softmax(logits, axis=1)
        loss = nll_loss(log_probs, yb)
        losses.append(float(loss.data))
        all_logits.append(logits.data)
        all_targets.append(yb)

    logits_cat = np.concatenate(all_logits, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)
    return float(np.mean(losses)), accuracy_from_logits(logits_cat, targets_cat)


def save_learning_curve(rows: list[EpochMetrics], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for r in rows:
            writer.writerow([r.epoch, r.train_loss, r.train_acc, r.val_loss, r.val_acc])
