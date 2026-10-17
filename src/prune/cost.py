from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from engine.tensor import Tensor
from nn.modules import Linear, MLP


@dataclass
class CostSummary:
    dense_params: int
    active_params: int
    dense_macs_per_example: int
    sparse_macs_per_example: int
    dense_ms_per_batch: float
    sparse_ms_per_batch: float


def _linear_layers(model: MLP) -> list[Linear]:
    return [layer for layer in model.net.layers if isinstance(layer, Linear)]


def count_dense_and_active_params(model: MLP) -> tuple[int, int]:
    dense = 0
    active = 0
    for layer in _linear_layers(model):
        w_total = int(np.prod(layer.weight.data.shape))
        dense += w_total
        if layer.weight.mask is None:
            active += w_total
        else:
            active += int(np.count_nonzero(layer.weight.mask))
    return dense, active


def estimate_macs_per_example(model: MLP) -> tuple[int, int]:
    dense = 0
    sparse = 0
    for layer in _linear_layers(model):
        in_features, out_features = layer.weight.data.shape
        dense += int(in_features * out_features)
        if layer.weight.mask is None:
            sparse += int(in_features * out_features)
        else:
            sparse += int(np.count_nonzero(layer.weight.mask))
    return dense, sparse


def sparse_forward_numpy(model: MLP, x: np.ndarray) -> np.ndarray:
    activ = x
    if not np.isfinite(activ).all():
        raise FloatingPointError("Non-finite input passed to sparse forward")
    layers = model.net.layers
    for layer in layers:
        if isinstance(layer, Linear):
            w = layer.weight.data
            b = layer.bias.data
            mask = layer.weight.mask
            batch = activ.shape[0]
            out = np.zeros((batch, w.shape[1]), dtype=np.float64)

            if mask is None:
                with np.errstate(all="ignore"):
                    out = activ @ w
            else:
                for j in range(w.shape[1]):
                    col_mask = mask[:, j] != 0.0
                    if not np.any(col_mask):
                        continue
                    with np.errstate(all="ignore"):
                        out[:, j] = activ[:, col_mask] @ w[col_mask, j]

            out += b
            if not np.isfinite(out).all():
                raise FloatingPointError("Non-finite values produced in sparse forward")
            activ = out
        else:
            # ReLU layer in this project.
            activ = np.maximum(activ, 0.0)

    return activ


def benchmark_forward_ms(model: MLP, x: np.ndarray, repeats: int = 20) -> tuple[float, float]:
    if repeats <= 0:
        raise ValueError("repeats must be >= 1")

    # Warmup
    _ = model(Tensor(x, requires_grad=False))
    _ = sparse_forward_numpy(model, x)

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model(Tensor(x, requires_grad=False))
    dense_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    t1 = time.perf_counter()
    for _ in range(repeats):
        _ = sparse_forward_numpy(model, x)
    sparse_ms = (time.perf_counter() - t1) * 1000.0 / repeats

    return float(dense_ms), float(sparse_ms)


def summarize_cost(model: MLP, x_sample: np.ndarray, repeats: int = 20) -> CostSummary:
    dense_params, active_params = count_dense_and_active_params(model)
    dense_macs, sparse_macs = estimate_macs_per_example(model)
    dense_ms, sparse_ms = benchmark_forward_ms(model, x_sample, repeats=repeats)

    return CostSummary(
        dense_params=dense_params,
        active_params=active_params,
        dense_macs_per_example=dense_macs,
        sparse_macs_per_example=sparse_macs,
        dense_ms_per_batch=dense_ms,
        sparse_ms_per_batch=sparse_ms,
    )
