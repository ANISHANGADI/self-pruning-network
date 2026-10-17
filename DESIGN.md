# DESIGN

## 1) Autodiff Engine Design

### Core representation
- `Tensor` wraps a NumPy array and stores:
  - `data`
  - `grad`
  - `_children` (parents in graph)
  - `_backward` closure
- Reverse-mode autodiff is implemented by topological traversal in `backward()`.

### Correctness details
- Gradients accumulate when a node is used multiple times.
- Broadcasting is handled with an explicit `_unbroadcast` reduction step.
- Matrix multiplication checks finite values in forward and backward paths.

### Masked-weight semantics
- `Parameter` can own a binary mask.
- Forward uses `weight.masked()` so masked weights are hard-zero in compute.
- Mask updates apply in-place to both weights and gradients.
- Adam state tracks prior mask and resets moments when a coordinate revives, avoiding stale-momentum corruption.

## 2) Model + Optimization

### Network
- MLP with `Linear + ReLU` blocks.
- Default architecture: `784 -> 256 -> 128 -> 10`.

### Initialization
- He-normal for linear weights (`sqrt(2 / fan_in)`) for ReLU stability.
- Zero bias initialization.

### Optimizer
- Adam with optional weight decay.
- Mask-aware update logic:
  - masked coordinates receive zero gradient updates
  - moments are masked after step
  - revived coordinates reset moments

## 3) Pruning Design

### Criteria
- Magnitude baseline: `|w|`
- Saliency criterion: `|w * grad|`
  - First-order Taylor approximation motivates this:
    - Removing a weight corresponds to perturbation `delta_w = -w`
    - First-order loss change magnitude is approximately `|grad * delta_w| = |w * grad|`

### Schedule
- Gradual cubic sparsity ramp:
  - `s(t) = s_f + (s_i - s_f) * (1 - t/T)^3`
- Global pruning across all masked layers.
- Pruning events occur every `prune_interval` eligible optimizer steps.
- Final eligible step always triggers pruning event, so target is not skipped when interval is large.

### Honest masking
- Pruned weights are exactly zero (`mask=0`), not merely small values.
- Forward computation uses mask during matmul.

## 4) Evidence and Measurement Strategy

### Pareto evidence
- Run sweep across target sparsities and seeds.
- Record per run:
  - achieved sparsity
  - validation/test loss and accuracy
  - active/dense parameter counts
  - dense and sparse MAC estimates
  - dense vs sparse-aware forward latency

### Real compute evidence
- Dense MACs per example: sum of `fan_in * fan_out` over linear layers.
- Sparse MACs per example: count of active (unmasked) connections.
- Sparse-aware forward timing uses explicit masked-connection evaluation, not dense-times-zero.
- This implementation prioritizes honest operation counting; wall-clock may still favor dense BLAS at this scale.

### Baseline comparison
- Compare `magnitude` vs `saliency` at matched target sparsity and identical seeds.
- Aggregate mean/std across seeds.

### Falsifiable claim format
- Claim is emitted by `scripts/summarize_sweep.py` at a chosen sparsity.
- Example format:
  - "At target sparsity 0.90, saliency test accuracy mean = X, magnitude = Y, delta = Z, over N seeds."

## 5) Reproducibility

- Fixed random seeds used for data split and batch order.
- Scripts:
  - `scripts/train_dense_baseline.py`
  - `scripts/train_self_pruning.py`
  - `scripts/run_pruning_sweep.py`
  - `scripts/plot_pareto.py`
  - `scripts/summarize_sweep.py`
- Tests include:
  - grad checks
  - masking semantics and optimizer-state correctness
  - pruning schedule and criterion behavior
