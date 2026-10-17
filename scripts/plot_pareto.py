from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot sparsity-accuracy Pareto from pruning sweep CSV")
    p.add_argument("--csv", type=str, default="results/pruning_sweep.csv")
    p.add_argument("--out", type=str, default="plots/sparsity_accuracy_pareto.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    grouped: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            criterion = row["criterion"]
            target = float(row["target_sparsity"])
            grouped[(criterion, target)].append(
                {
                    "achieved_sparsity": float(row["achieved_sparsity"]),
                    "test_acc": float(row["test_acc"]),
                }
            )

    summary: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (criterion, _target), rows in grouped.items():
        spars = np.array([r["achieved_sparsity"] for r in rows], dtype=np.float64)
        acc = np.array([r["test_acc"] for r in rows], dtype=np.float64)
        summary[criterion].append((float(np.mean(spars)), float(np.mean(acc)), float(np.std(acc))))

    fig, ax = plt.subplots(figsize=(7, 5))
    for criterion, pts in summary.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        x = [p[0] for p in pts_sorted]
        y = [p[1] for p in pts_sorted]
        yerr = [p[2] for p in pts_sorted]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=criterion)

    ax.set_title("Sparsity-Accuracy Pareto")
    ax.set_xlabel("Achieved Sparsity")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved_plot={out_path}")


if __name__ == "__main__":
    main()
