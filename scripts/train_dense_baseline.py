from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from data.mnist import load_mnist
from engine.tensor import Tensor, log_softmax, nll_loss
from nn.modules import MLP
from optim.adam import Adam
from utils.training import EpochMetrics, evaluate_model, iterate_minibatches, save_learning_curve, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dense MNIST baseline with custom autodiff")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="256,128")
    parser.add_argument("--out-csv", type=str, default="results/dense_learning_curve_seed7.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_dims = [int(x.strip()) for x in args.hidden.split(",") if x.strip()]

    set_seed(args.seed)
    data = load_mnist(seed=args.seed, val_size=5000)

    model = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10, use_mask=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    history: list[EpochMetrics] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses: list[float] = []
        train_logits: list[np.ndarray] = []
        train_targets: list[np.ndarray] = []

        for xb, yb in iterate_minibatches(data["x_train"], data["y_train"], args.batch_size, rng, shuffle=True):
            optimizer.zero_grad()
            logits = model(Tensor(xb, requires_grad=False))
            log_probs = log_softmax(logits, axis=1)
            loss = nll_loss(log_probs, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.data))
            train_logits.append(logits.data)
            train_targets.append(yb)

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(np.argmax(np.concatenate(train_logits, axis=0), axis=1) == np.concatenate(train_targets, axis=0)))
        val_loss, val_acc = evaluate_model(model, data["x_val"], data["y_val"])

        history.append(EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc))

        dt = time.time() - t0
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time_s={dt:.2f}"
        )

    test_loss, test_acc = evaluate_model(model, data["x_test"], data["y_test"])
    print(f"final_test_loss={test_loss:.4f} final_test_acc={test_acc:.4f}")

    out_csv = os.path.join(ROOT, args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    save_learning_curve(history, out_csv)
    print(f"saved_learning_curve={out_csv}")


if __name__ == "__main__":
    main()
