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
from prune.cost import summarize_cost
from prune.methods import apply_global_pruning_step, cubic_sparsity, sparsity_from_params
from utils.training import evaluate_model, iterate_minibatches, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pruning sweeps across sparsity targets and seeds")
    p.add_argument("--seeds", type=str, default="7,13,21")
    p.add_argument("--targets", type=str, default="0.0,0.5,0.75,0.9,0.95")
    p.add_argument("--criteria", type=str, default="magnitude,saliency")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=str, default="256,128")
    p.add_argument("--prune-start-epoch", type=int, default=2)
    p.add_argument("--prune-end-epoch", type=int, default=7)
    p.add_argument("--prune-interval", type=int, default=100)
    p.add_argument("--timing-batch-size", type=int, default=512)
    p.add_argument("--timing-repeats", type=int, default=10)
    p.add_argument("--out", type=str, default="results/pruning_sweep.csv")
    return p.parse_args()


def _count_events(num_batches: int, start_epoch: int, end_epoch: int, epochs: int, interval: int) -> tuple[int, int]:
    if interval <= 0:
        raise ValueError("prune_interval must be >= 1")
    if start_epoch > end_epoch:
        return 0, 0

    active_epochs = max(0, min(end_epoch, epochs) - max(start_epoch, 1) + 1)
    eligible_steps = active_epochs * num_batches
    if eligible_steps == 0:
        return 0, 0

    events = (eligible_steps + interval - 1) // interval
    return events, eligible_steps


def train_once(
    seed: int,
    criterion: str,
    target_sparsity: float,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dims: list[int],
    prune_start_epoch: int,
    prune_end_epoch: int,
    prune_interval: int,
):
    set_seed(seed)
    data = load_mnist(seed=seed, val_size=5000)

    model = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10, use_mask=True)
    optimizer = Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed)
    steps_per_epoch = (data["x_train"].shape[0] + batch_size - 1) // batch_size
    total_events, total_eligible = _count_events(steps_per_epoch, prune_start_epoch, prune_end_epoch, epochs, prune_interval)

    events_done = 0
    eligible_seen = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(data["x_train"], data["y_train"], batch_size, rng, shuffle=True):
            optimizer.zero_grad()
            logits = model(Tensor(xb, requires_grad=False))
            log_probs = log_softmax(logits, axis=1)
            loss = nll_loss(log_probs, yb)
            loss.backward()
            optimizer.step()

            if criterion != "none" and target_sparsity > 0.0 and prune_start_epoch <= epoch <= prune_end_epoch and total_events > 0:
                eligible_seen += 1
                should_prune = (eligible_seen % prune_interval == 0) or (eligible_seen == total_eligible)
                if should_prune and events_done < total_events:
                    events_done += 1
                    target = cubic_sparsity(events_done, total_events, target_sparsity)
                    apply_global_pruning_step(model.parameters(), criterion=criterion, target_sparsity=target)

    train_time_s = float(time.time() - t0)

    val_loss, val_acc = evaluate_model(model, data["x_val"], data["y_val"])
    test_loss, test_acc = evaluate_model(model, data["x_test"], data["y_test"])
    sparsity, active_params, total_params = sparsity_from_params(model.parameters())

    return {
        "model": model,
        "train_time_s": train_time_s,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "achieved_sparsity": sparsity,
        "active_params": active_params,
        "total_params": total_params,
        "data": data,
    }


def main() -> None:
    args = parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    criteria = [x.strip() for x in args.criteria.split(",") if x.strip()]
    hidden_dims = [int(x.strip()) for x in args.hidden.split(",") if x.strip()]

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows: list[dict[str, float | int | str]] = []

    for criterion in criteria:
        for target in targets:
            for seed in seeds:
                run = train_once(
                    seed=seed,
                    criterion=criterion,
                    target_sparsity=target,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    hidden_dims=hidden_dims,
                    prune_start_epoch=args.prune_start_epoch,
                    prune_end_epoch=args.prune_end_epoch,
                    prune_interval=args.prune_interval,
                )

                model = run["model"]
                x_sample = run["data"]["x_test"][: args.timing_batch_size]
                cost = summarize_cost(model, x_sample=x_sample, repeats=args.timing_repeats)

                row = {
                    "criterion": criterion,
                    "target_sparsity": target,
                    "seed": seed,
                    "achieved_sparsity": run["achieved_sparsity"],
                    "val_acc": run["val_acc"],
                    "test_acc": run["test_acc"],
                    "val_loss": run["val_loss"],
                    "test_loss": run["test_loss"],
                    "train_time_s": run["train_time_s"],
                    "active_params": cost.active_params,
                    "dense_params": cost.dense_params,
                    "dense_macs_per_example": cost.dense_macs_per_example,
                    "sparse_macs_per_example": cost.sparse_macs_per_example,
                    "dense_ms_per_batch": cost.dense_ms_per_batch,
                    "sparse_ms_per_batch": cost.sparse_ms_per_batch,
                }
                rows.append(row)

                print(
                    f"criterion={criterion} target={target:.2f} seed={seed} "
                    f"acc={run['test_acc']:.4f} sparsity={run['achieved_sparsity']:.4f} "
                    f"active={cost.active_params}/{cost.dense_params}"
                )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "criterion",
                "target_sparsity",
                "seed",
                "achieved_sparsity",
                "val_acc",
                "test_acc",
                "val_loss",
                "test_loss",
                "train_time_s",
                "active_params",
                "dense_params",
                "dense_macs_per_example",
                "sparse_macs_per_example",
                "dense_ms_per_batch",
                "sparse_ms_per_batch",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"saved_sweep={out_path}")


if __name__ == "__main__":
    main()
