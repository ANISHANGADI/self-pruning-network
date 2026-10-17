import numpy as np

from nn.parameter import Parameter
from prune.methods import apply_global_pruning_step, cubic_sparsity, sparsity_from_params


def test_cubic_sparsity_endpoints_and_monotonicity():
    total = 20
    vals = [cubic_sparsity(s, total, final_sparsity=0.9, initial_sparsity=0.0) for s in range(total + 1)]

    assert vals[0] == 0.0
    assert abs(vals[-1] - 0.9) < 1e-12
    for i in range(len(vals) - 1):
        assert vals[i + 1] >= vals[i]


def test_global_magnitude_pruning_hits_target_sparsity():
    p1 = Parameter(np.array([1.0, 2.0, 3.0, 4.0]), use_mask=True)
    p2 = Parameter(np.array([5.0, 6.0, 7.0, 8.0]), use_mask=True)

    stats = apply_global_pruning_step([p1, p2], criterion="magnitude", target_sparsity=0.5)

    assert stats.total_params == 8
    assert stats.active_params == 4
    assert stats.pruned_this_step == 4
    assert abs(stats.achieved_sparsity - 0.5) < 1e-12

    # Lowest magnitudes should be pruned globally: 1,2,3,4
    assert np.allclose(p1.mask, np.zeros_like(p1.mask))
    assert np.allclose(p2.mask, np.ones_like(p2.mask))


def test_saliency_pruning_uses_weight_gradient_product():
    p = Parameter(np.array([10.0, 1.0, 2.0, 3.0]), use_mask=True)
    p.tensor.grad = np.array([0.01, 100.0, 1.0, 1.0], dtype=np.float64)

    # saliency scores: [0.1, 100, 2, 3], so prune the smallest (index 0).
    stats = apply_global_pruning_step([p], criterion="saliency", target_sparsity=0.25)

    assert stats.pruned_this_step == 1
    assert p.mask[0] == 0.0
    assert p.mask[1] == 1.0
    assert p.mask[2] == 1.0
    assert p.mask[3] == 1.0

    sparsity, active, total = sparsity_from_params([p])
    assert total == 4
    assert active == 3
    assert abs(sparsity - 0.25) < 1e-12
