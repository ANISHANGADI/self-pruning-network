# Self-Pruning Network (NumPy + custom autodiff)

This repository implements a neural network and pruning pipeline from scratch in Python + NumPy.

## Important restrictions
- `scikit-learn` is used only for dataset loading and splitting.
- No PyTorch/TensorFlow/JAX/Keras/autodiff libraries are used for model training or gradients.
- Backpropagation, optimizer, pruning logic, and training loop are custom code.

## Project status
- [x] Phase 1 started: reverse-mode autodiff engine and gradient checks
- [ ] Phase 2: mask semantics + optimizer correctness tests
- [x] Phase 2: mask semantics + optimizer correctness tests
- [x] Phase 3: MLP + Adam + dense training baseline
- [ ] Phase 4: self-pruning criteria and schedules
- [ ] Phase 5: Pareto evidence, sparse compute evidence, and falsifiable claim

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
