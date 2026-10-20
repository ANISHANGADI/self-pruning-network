from __future__ import annotations

import argparse
import csv
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
from prune.methods import apply_global_pruning_step, cubic_sparsity, sparsity_from_params
from prune.methods import capture_importance, pruning_events, assert_pruning_budget
from utils.training import evaluate_model, iterate_minibatches, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST model with gradual self-pruning")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="256,128")
    parser.add_argument("--criterion", type=str, default="saliency", choices=["magnitude", "saliency"])
    parser.add_argument("--target-sparsity", type=float, default=0.9)
    parser.add_argument("--prune-start-epoch", type=int, default=2)
    parser.add_argument("--prune-end-epoch", type=int, default=10)
    parser.add_argument("--prune-interval", type=int, default=100)
    parser.add_argument("--out-csv", type=str, default="results/self_pruning_learning_curve_seed7.csv")
    return parser.parse_args()


def _count_pruning_events(num_batches: int, prune_start_epoch: int, prune_end_epoch: int, epochs: int, prune_interval: int) -> int:
    return pruning_events(num_batches, prune_start_epoch, prune_end_epoch, epochs, prune_interval)[0]


def _count_eligible_steps(num_batches: int, prune_start_epoch: int, prune_end_epoch: int, epochs: int) -> int:
    return pruning_events(num_batches, prune_start_epoch, prune_end_epoch, epochs, 1)[1]


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.target_sparsity <= 1.0):
        raise ValueError("target_sparsity must be in [0, 1]")
    if args.batch_size < 1 or args.lr <= 0:
        raise ValueError("batch size and learning rate must be positive")
    pruning_events(1, args.prune_start_epoch, args.prune_end_epoch, args.epochs, args.prune_interval)

    hidden_dims = [int(x.strip()) for x in args.hidden.split(",") if x.strip()]
    set_seed(args.seed)

    data = load_mnist(seed=args.seed, val_size=5000)
    model = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10, use_mask=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    steps_per_epoch = (data["x_train"].shape[0] + args.batch_size - 1) // args.batch_size
    total_prune_events = _count_pruning_events(
        steps_per_epoch,
        args.prune_start_epoch,
        args.prune_end_epoch,
        args.epochs,
        args.prune_interval,
    )
    total_eligible_steps = _count_eligible_steps(
        steps_per_epoch,
        args.prune_start_epoch,
        args.prune_end_epoch,
        args.epochs,
    )

    history_rows: list[dict[str, float | int]] = []
    prune_events_done = 0
    eligible_seen = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses: list[float] = []
        train_logits: list[np.ndarray] = []
        train_targets: list[np.ndarray] = []
        pruned_this_epoch = 0

        for xb, yb in iterate_minibatches(data["x_train"], data["y_train"], args.batch_size, rng, shuffle=True):
            optimizer.zero_grad()
            logits = model(Tensor(xb, requires_grad=False))
            log_probs = log_softmax(logits, axis=1)
            loss = nll_loss(log_probs, yb)
            loss.backward()
            scores = None
            if args.prune_start_epoch <= epoch <= args.prune_end_epoch:
                eligible_seen += 1
                if eligible_seen % args.prune_interval == 0 or eligible_seen == total_eligible_steps:
                    scores = capture_importance(model.parameters(), args.criterion)
            optimizer.step()

            train_losses.append(float(loss.data))
            train_logits.append(logits.data)
            train_targets.append(yb)

            if scores is not None:
                prune_events_done += 1
                target = cubic_sparsity(prune_events_done, total_prune_events, args.target_sparsity)
                stats = apply_global_pruning_step(model.parameters(), args.criterion, target, scores=scores)
                pruned_this_epoch += stats.pruned_this_step

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(np.argmax(np.concatenate(train_logits, axis=0), axis=1) == np.concatenate(train_targets, axis=0)))
        val_loss, val_acc = evaluate_model(model, data["x_val"], data["y_val"])
        sparsity, active_params, total_params = sparsity_from_params(model.parameters())

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "sparsity": sparsity,
                "active_params": active_params,
                "total_params": total_params,
                "pruned_this_epoch": pruned_this_epoch,
            }
        )

        dt = time.time() - t0
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} sparsity={sparsity:.4f} "
            f"pruned={pruned_this_epoch} time_s={dt:.2f}"
        )

    assert_pruning_budget(model.parameters(), args.target_sparsity)
    test_loss, test_acc = evaluate_model(model, data["x_test"], data["y_test"])
    final_sparsity, final_active, final_total = sparsity_from_params(model.parameters())
    print(
        f"final_test_loss={test_loss:.4f} final_test_acc={test_acc:.4f} "
        f"final_sparsity={final_sparsity:.4f} active={final_active}/{final_total}"
    )

    out_csv = os.path.join(ROOT, args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "sparsity",
                "active_params",
                "total_params",
                "pruned_this_epoch",
            ],
        )
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)

    print(f"saved_learning_curve={out_csv}")


if __name__ == "__main__":
    main()
