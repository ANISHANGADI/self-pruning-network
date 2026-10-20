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
- Differentiable matrix multiplication supports the 2D matrices used by the MLP;
  vector and batched matmul gradients are outside this implementation's scope.

### Masked-weight semantics
- `Parameter` can own a binary mask.
- Forward uses `weight.masked()` so masked weights are hard-zero in compute.
- Mask updates apply in-place to both weights and gradients.
- Adam state tracks prior mask and resets moments when a coordinate revives, avoiding stale-momentum corruption.
- For a fixed mask, `W_eff = W * M` gives `dL/dW = dL/dW_eff * M`.
  The effective-weight sensitivity may be nonzero at a masked position, while
  the trainable-weight gradient is zero. Do not treat that zero gradient as
  evidence that a disconnected connection could never be useful again.
- Supported training uses a fresh graph each batch, one backward call, and mask
  changes only after backward. Do not mutate masks/weights during a live graph.

## 2) Model + Optimization

### Network
- MLP with `Linear + ReLU` blocks.
- Default architecture: `784 -> 256 -> 128 -> 10`.

### Initialization
- He-normal weights have standard deviation `sqrt(2 / fan_in)`. This compensates
  approximately for ReLU discarding negative activations, helping preserve signal
  scale through hidden layers. The same initialization is used for the output
  layer for simplicity, although it has no ReLU and the argument does not apply
  there directly.
- Biases start at zero; random weights break symmetry between neurons.

### Optimizer
- Adam with optional L2 regularization added to the gradient, not decoupled AdamW
  weight decay. The reported experiments use no weight decay.
- Mask-aware update logic:
  - masked coordinates receive zero gradient updates
  - moments are masked after step
  - revived coordinates reset moments

Revival policy: reset only `m` and `v`, retaining the optimizer's global `t`.
With fresh gradient `g` at global step `t`, the revived entry uses
`m_hat = (1-beta1)*g/(1-beta1**t)` and
`v_hat = (1-beta2)*g*g/(1-beta2**t)`. This is not a new optimizer at step 1;
late revival can have a different effective step size. A numerical regression
test checks the exact step-4 update and preserves an unaffected coordinate.
Snapshots detect transitions between observed optimizer steps; a 1->0->1 toggle
entirely between observations is not detected. A future regrowth system needs
an explicit transition/reset API if it permits that lifecycle. Current pruning
never regrows connections, so revival is tested support, not an experiment feature.

## 3) Pruning Design

### Criteria
- Magnitude baseline: `|w|`
- Saliency criterion: `|w * grad|`
  - First-order Taylor approximation motivates this:
    - Removing a weight corresponds to perturbation `delta_w = -w`
    - First-order loss change magnitude is approximately `|grad * delta_w| = |w * grad|`
- Scores for both methods are copied after backward and before the Adam step,
  and masks are applied after Adam using those snapshots. Weight and gradient
  therefore refer to the same pre-update point. Scores are a local proxy,
  not an exact prediction of the post-update removal loss.
- Absolute Taylor change measures sensitivity, not guaranteed loss increase:
  the signed estimate is `-w * grad`. Removing many weights is a finite change,
  higher-order interactions matter, and a single minibatch gradient is noisy.
  We do not assume saliency must outperform magnitude.

### Schedule
- Gradual cubic sparsity ramp:
  - `s(t) = s_f + (s_i - s_f) * (1 - t/T)^3`
- Global pruning across all masked layers.
- Pruning events occur every `prune_interval` eligible optimizer steps.
- Final eligible step always triggers pruning event, so target is not skipped when interval is large.
- Require `1 <= start <= end <= epochs` and positive interval/batch size.
- Global weight budget is `floor((1-s)*N)` using the decimal spelling of `s`;
  biases are not pruned. At 95%, 234,752 weights allow 11,737 active connections,
  not 11,738. Each trainer explicitly verifies the final budget.
- Warmup permits initial learning; cubic decay reduces pruning increments near
  the end, and remaining epochs allow recovery. This is a design choice, not a
  demonstrated optimal schedule; no schedule ablation is claimed.

### Honest masking
- Pruned weights are exactly zero (`mask=0`), not merely small values.
- Forward computation uses mask during matmul.
- Training still performs dense multiplication, including zeros. Hard masking
  provides correct values/gradients, not reduced dense training work or memory.

## 4) Evidence and Measurement Strategy

### Pareto evidence
- Run sweep across target sparsities and seeds.
- MNIST pixels are scaled to [0, 1]. The original 60,000 training examples are
  split into 55,000 training and 5,000 validation examples using each run's seed;
  the fixed 10,000-example test set is held out from training.
- Record per run:
  - achieved sparsity
  - validation/test loss and accuracy
  - active/dense parameter counts
  - dense and sparse MAC estimates
  - dense vs sparse-aware forward latency
- Accuracy counts correct predictions over all examples. Logged losses average
  minibatch means without weighting the smaller final batch; they are not exact
  dataset-mean losses. The Pareto comparison and claim use accuracy, not loss.

### Real compute evidence
- Dense MACs per example: sum of `fan_in * fan_out` over linear layers.
- Sparse MACs per example: count of active (unmasked) connections.
- Sparse-aware forward timing uses explicit masked-connection evaluation, not dense-times-zero.
- Connection MAC counts measure arithmetic work, not total runtime or storage.
  Dense BLAS is faster than the sparse implementation in the reported experiments.
- The dense baseline is also NumPy-only. Warmup checks sparse/dense output
  agreement; unit tests also compare both with the autodiff model, including
  zero/full/random masks, biases, and batch sizes 1 and 5.
- Timings include mask construction/scanning and sparse gathers, with one warmup
  and ten repetitions on a 512-example batch. Counts cover weight MACs only, not
  bias additions, activations, indexing, or allocation. No compressed storage is
  implemented. At 90%, measured sparse/dense ratios are 13.07x (magnitude) and
  12.57x (saliency): lower arithmetic count, but slower inference.

### Baseline comparison
- Compare `magnitude` vs `saliency` at matched target sparsity and identical seeds.
- Aggregate mean/std across seeds.
- The reference sweep gives 90%-sparse magnitude accuracy 0.980967
  versus saliency 0.974433. Paired saliency-minus-magnitude differences are
  -0.0053, -0.0063, -0.0080 (mean -0.006533, sample std 0.001365).
  All three favor magnitude; they do not prove a universal ranking. Seed changes
  affect initialization, batch order, and validation split together. Summary and
  plot std use ddof=0 and describe observed variation, not confidence intervals.
  Further tuning must use validation data, not these test results.

### Falsifiable claim
At 90% target sparsity, magnitude pruning achieves 98.0967% mean test accuracy
versus 97.4433% for saliency across seeds 7, 13, and 21, a difference of 0.6533
percentage points in favor of magnitude. Both retain 23,475 of 234,752 weights.
The claim is supported by [raw results](results/pruning_sweep.csv) and the
[summary table](results/pruning_summary.csv), generated by
[scripts/summarize_sweep.py](scripts/summarize_sweep.py).

## 5) Reproducibility

- Fixed random seeds used for data split and batch order.
- Reference measurement environment: Python 3.11.17 on macOS 26.7 ARM64.
  Dependencies are recorded in [requirements-lock.txt](requirements-lock.txt).
  All 64 tests pass in this environment. Hardware and numerical-library changes
  can affect timings and numerical results; fixed seeds do not guarantee
  bitwise-identical reproduction.
- Reference sweep: 8 epochs, batch 128, learning rate 0.001, hidden 256/128,
  pruning epochs 2-7, interval 100, seeds 7/13/21, targets 0/0.5/0.75/0.9/0.95,
  criteria magnitude/saliency, timing batch 512 and 10 repeats. README contains
  the runnable commands. Prior evidence remains available in git history.
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
  - pre-update score capture and final-budget enforcement in both trainers
  - NumPy dense/sparse inference equivalence against the autodiff model
