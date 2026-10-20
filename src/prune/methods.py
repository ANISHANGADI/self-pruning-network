from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

import numpy as np

from nn.parameter import Parameter


@dataclass
class PruneStats:
    target_sparsity: float
    achieved_sparsity: float
    pruned_this_step: int
    active_params: int
    total_params: int


def cubic_sparsity(step: int, total_steps: int, final_sparsity: float, initial_sparsity: float = 0.0) -> float:
    if not (0.0 <= initial_sparsity <= 1.0):
        raise ValueError("initial_sparsity must be in [0, 1]")
    if not (0.0 <= final_sparsity <= 1.0):
        raise ValueError("final_sparsity must be in [0, 1]")
    if final_sparsity < initial_sparsity:
        raise ValueError("final_sparsity must be >= initial_sparsity")

    if total_steps <= 0:
        return final_sparsity

    clipped_step = min(max(step, 0), total_steps)
    progress = clipped_step / total_steps
    return final_sparsity + (initial_sparsity - final_sparsity) * ((1.0 - progress) ** 3)


def masked_parameters(params: list[Parameter]) -> list[Parameter]:
    return [p for p in params if p.mask is not None]


def sparsity_from_params(params: list[Parameter]) -> tuple[float, int, int]:
    masked = masked_parameters(params)
    if not masked:
        return 0.0, 0, 0

    total = int(sum(np.prod(p.mask.shape) for p in masked if p.mask is not None))
    active = int(sum(np.count_nonzero(p.mask) for p in masked if p.mask is not None))
    sparsity = 1.0 - (active / total)
    return float(sparsity), active, total


def _importance_for_param(param: Parameter, criterion: str) -> np.ndarray:
    if criterion == "magnitude":
        return np.abs(param.data)
    if criterion == "saliency":
        if param.grad is None:
            raise ValueError("saliency pruning requires gradients to be available")
        return np.abs(param.data * param.grad)
    raise ValueError(f"Unsupported pruning criterion: {criterion}")


def active_budget(total: int, target_sparsity: float) -> int:
    if total < 0 or not (0.0 <= target_sparsity <= 1.0):
        raise ValueError("Invalid total or target sparsity")
    return int(((Decimal(1) - Decimal(str(target_sparsity))) * total).to_integral_value(rounding=ROUND_FLOOR))


def pruning_events(num_batches: int, start_epoch: int, end_epoch: int, epochs: int, interval: int) -> tuple[int, int]:
    if num_batches < 1 or interval < 1 or not (1 <= start_epoch <= end_epoch <= epochs):
        raise ValueError("Require positive batches/interval and 1 <= prune start <= prune end <= epochs")
    eligible = (end_epoch - start_epoch + 1) * num_batches
    return (eligible + interval - 1) // interval, eligible


def capture_importance(params: list[Parameter], criterion: str) -> dict[int, np.ndarray]:
    return {id(param): _importance_for_param(param, criterion).copy() for param in masked_parameters(params)}


def assert_pruning_budget(params: list[Parameter], target_sparsity: float) -> None:
    _, active, total = sparsity_from_params(params)
    if active > active_budget(total, target_sparsity):
        raise RuntimeError("Final active connection count exceeds pruning budget")


def apply_global_pruning_step(params: list[Parameter], criterion: str, target_sparsity: float, scores: dict[int, np.ndarray] | None = None) -> PruneStats:
    if not (0.0 <= target_sparsity <= 1.0):
        raise ValueError("target_sparsity must be in [0, 1]")

    masked = masked_parameters(params)
    if not masked:
        return PruneStats(target_sparsity, 0.0, 0, 0, 0)

    total_params = int(sum(np.prod(p.mask.shape) for p in masked if p.mask is not None))
    desired_active = active_budget(total_params, target_sparsity)

    active_coords: list[tuple[Parameter, np.ndarray, np.ndarray]] = []
    active_scores: list[np.ndarray] = []

    current_active = 0
    for p in masked:
        assert p.mask is not None
        active_mask = p.mask > 0.0
        active_idx = np.flatnonzero(active_mask.ravel())
        current_active += int(active_idx.size)
        if active_idx.size == 0:
            continue

        importance = _importance_for_param(p, criterion) if scores is None else scores[id(p)]
        if importance.shape != p.data.shape or not np.isfinite(importance).all():
            raise ValueError("Importance scores must match parameter shape and be finite")
        active_coords.append((p, active_idx, active_mask))
        active_scores.append(importance.ravel()[active_idx])

    if desired_active >= current_active:
        achieved, active, total = sparsity_from_params(params)
        return PruneStats(target_sparsity, achieved, 0, active, total)

    prune_count = current_active - desired_active
    all_scores = np.concatenate(active_scores, axis=0)

    local_prune = np.argpartition(all_scores, prune_count - 1)[:prune_count]
    prune_selector = np.zeros_like(all_scores, dtype=bool)
    prune_selector[local_prune] = True

    offset = 0
    pruned = 0
    for p, active_idx, _ in active_coords:
        block = prune_selector[offset : offset + active_idx.size]
        offset += active_idx.size
        if not np.any(block):
            continue

        assert p.mask is not None
        flat_mask = p.mask.ravel()
        to_prune_idx = active_idx[block]
        flat_mask[to_prune_idx] = 0.0
        p.mask = flat_mask.reshape(p.mask.shape)
        p.apply_mask_inplace()
        pruned += int(to_prune_idx.size)

    achieved_sparsity, active_params, total = sparsity_from_params(params)
    return PruneStats(target_sparsity, achieved_sparsity, pruned, active_params, total)
