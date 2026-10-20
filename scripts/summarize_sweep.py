from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize pruning sweep results and emit a claim-ready table")
    p.add_argument("--csv", type=str, default="results/pruning_sweep.csv")
    p.add_argument("--out", type=str, default="results/pruning_summary.csv")
    p.add_argument("--claim-sparsity", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    grouped: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[(row["criterion"], float(row["target_sparsity"]))].append(
                {
                    "seed": int(row["seed"]),
                    "achieved_sparsity": float(row["achieved_sparsity"]),
                    "test_acc": float(row["test_acc"]),
                    "val_acc": float(row["val_acc"]),
                    "dense_macs_per_example": float(row["dense_macs_per_example"]),
                    "sparse_macs_per_example": float(row["sparse_macs_per_example"]),
                    "dense_ms_per_batch": float(row["dense_ms_per_batch"]),
                    "sparse_ms_per_batch": float(row["sparse_ms_per_batch"]),
                }
            )

    summary_rows: list[dict[str, float | str | int]] = []
    for (criterion, target), rows in sorted(grouped.items(), key=lambda t: (t[0][0], t[0][1])):
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError("Duplicate seed in criterion/target group")
        achieved = np.array([r["achieved_sparsity"] for r in rows], dtype=np.float64)
        test_acc = np.array([r["test_acc"] for r in rows], dtype=np.float64)
        val_acc = np.array([r["val_acc"] for r in rows], dtype=np.float64)
        dense_macs = np.array([r["dense_macs_per_example"] for r in rows], dtype=np.float64)
        sparse_macs = np.array([r["sparse_macs_per_example"] for r in rows], dtype=np.float64)
        dense_ms = np.array([r["dense_ms_per_batch"] for r in rows], dtype=np.float64)
        sparse_ms = np.array([r["sparse_ms_per_batch"] for r in rows], dtype=np.float64)

        summary_rows.append(
            {
                "criterion": criterion,
                "target_sparsity": target,
                "n_seeds": len(rows),
                "achieved_sparsity_mean": float(np.mean(achieved)),
                "achieved_sparsity_std": float(np.std(achieved)),
                "test_acc_mean": float(np.mean(test_acc)),
                "test_acc_std": float(np.std(test_acc)),
                "val_acc_mean": float(np.mean(val_acc)),
                "val_acc_std": float(np.std(val_acc)),
                "sparse_macs_fraction": float(np.mean(sparse_macs / dense_macs)),
                "sparse_ms_over_dense_ms": float(np.mean(sparse_ms / dense_ms)),
            }
        )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "criterion",
                "target_sparsity",
                "n_seeds",
                "achieved_sparsity_mean",
                "achieved_sparsity_std",
                "test_acc_mean",
                "test_acc_std",
                "val_acc_mean",
                "val_acc_std",
                "sparse_macs_fraction",
                "sparse_ms_over_dense_ms",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"saved_summary={out_path}")

    claim_target = args.claim_sparsity
    lookup = {(r["criterion"], float(r["target_sparsity"])): r for r in summary_rows}
    mag = lookup.get(("magnitude", claim_target))
    sal = lookup.get(("saliency", claim_target))

    if mag is None or sal is None:
        print(
            "claim_status=unavailable "
            f"reason=missing_rows_for_target_{claim_target} "
            "needed_criteria=magnitude_and_saliency"
        )
        return

    diff = float(sal["test_acc_mean"] - mag["test_acc_mean"])
    magnitude_by_seed = {row["seed"]: row for row in grouped[("magnitude", claim_target)]}
    saliency_by_seed = {row["seed"]: row for row in grouped[("saliency", claim_target)]}
    if magnitude_by_seed.keys() != saliency_by_seed.keys():
        raise ValueError("Paired comparison requires identical seed sets")
    deltas = []
    for seed in sorted(magnitude_by_seed):
        magnitude = magnitude_by_seed[seed]
        saliency = saliency_by_seed[seed]
        if not np.isclose(magnitude["achieved_sparsity"], saliency["achieved_sparsity"], rtol=0, atol=1e-12):
            raise ValueError("Paired comparison requires matched achieved sparsity")
        delta = saliency["test_acc"] - magnitude["test_acc"]
        deltas.append(delta)
        print(f"paired_seed={seed} delta_accuracy={delta:.6f}")
    sample_std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    print(f"paired_delta_mean={np.mean(deltas):.6f} paired_delta_sample_std={sample_std:.6f}")
    print("uncertainty=descriptive_seed_variation_not_a_universal_superiority_claim; summary_std=population_ddof_0")
    print(
        "claim="
        f"at_target_{claim_target:.2f}, saliency_test_acc_mean={sal['test_acc_mean']:.4f}, "
        f"magnitude_test_acc_mean={mag['test_acc_mean']:.4f}, delta={diff:.4f}, "
        f"n_seeds={int(sal['n_seeds'])}"
    )


if __name__ == "__main__":
    main()
