# Self-Pruning Network (NumPy + custom autodiff)

This repository implements a neural network and pruning pipeline from scratch in Python + NumPy.

## Important restrictions
- `scikit-learn` is used only for dataset loading and splitting.
- No PyTorch/TensorFlow/JAX/Keras/autodiff libraries are used for model training or gradients.
- Backpropagation, optimizer, pruning logic, and training loop are custom code.

## Project status
- [x] Phase 1 started: reverse-mode autodiff engine and gradient checks
- [x] Phase 2: mask semantics + optimizer correctness tests
- [x] Phase 3: MLP + Adam + dense training baseline
- [x] Phase 4: self-pruning criteria and schedules
- [x] Phase 5: Pareto evidence, sparse compute evidence, and falsifiable claim

## Phase 3 baseline (MNIST)
- Architecture: `784 -> 256 -> 128 -> 10` with ReLU hidden layers
- Initialization: He-normal for linear weights, zero bias
- Optimizer: custom Adam
- Seed: `7`
- 5-epoch dense baseline result: `test_acc = 0.9777`, `test_loss = 0.0702`

### Reproduce baseline
```bash
python3 -m pytest -q
python3 scripts/train_dense_baseline.py --epochs 5 --batch-size 128 --lr 0.001 --seed 7 --out-csv results/dense_learning_curve_seed7.csv
python3 scripts/plot_learning_curve.py --csv results/dense_learning_curve_seed7.csv --out plots/dense_learning_curve_seed7.png
```

Notes:
- `fetch_openml` is configured with `parser="liac-arff"` to avoid a hard pandas dependency.
- Matrix multiplications explicitly validate finite outputs and gradients to catch true numerical failures.

## Phase 4 gradual self-pruning
- Criteria implemented: `magnitude` and `saliency` (|w * grad|)
- Schedule implemented: gradual cubic sparsity ramp with global mask updates
- Logging: epoch-wise sparsity, active parameter count, and pruned connections

### Reproduce self-pruning training
```bash
python3 -m pytest -q
python3 scripts/train_self_pruning.py --criterion saliency --target-sparsity 0.90 --epochs 12 --prune-start-epoch 2 --prune-end-epoch 10 --prune-interval 100 --out-csv results/self_pruning_learning_curve_seed7.csv
```

## Phase 5 evidence pipeline
Run a sweep across sparsity targets, criteria, and seeds:

```bash
python3 scripts/run_pruning_sweep.py --seeds 7,13,21 --targets 0.0,0.5,0.75,0.9,0.95 --criteria magnitude,saliency --epochs 8 --batch-size 128 --lr 0.001 --prune-start-epoch 2 --prune-end-epoch 7 --prune-interval 100 --out results/pruning_sweep.csv
```

Generate plots and summary table:

```bash
python3 scripts/plot_pareto.py --csv results/pruning_sweep.csv --out plots/sparsity_accuracy_pareto.png
python3 scripts/summarize_sweep.py --csv results/pruning_sweep.csv --out results/pruning_summary.csv --claim-sparsity 0.9
```

Evidence fields written by the sweep include:
- achieved sparsity and test accuracy per seed
- active parameter count vs dense parameter count
- dense MACs per example vs sparse MACs per example
- dense forward latency vs sparse-aware forward latency

Note:
- The sparse-aware forward path in this repository is an explicit NumPy implementation used for honest work accounting. It may reduce MACs and active parameters while still being slower wall-clock than dense BLAS matmul at this scale.
