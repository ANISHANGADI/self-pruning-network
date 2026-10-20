import numpy as np
import pytest
import importlib.util
from pathlib import Path
import sys

from nn.parameter import Parameter
from prune.methods import apply_global_pruning_step, cubic_sparsity, sparsity_from_params
from prune.methods import active_budget, capture_importance, pruning_events, assert_pruning_budget


def test_integer_budget_and_saved_scores():
    param = Parameter(np.arange(1.0, 8.0), use_mask=True)
    param.tensor.grad = np.ones(7)
    scores = capture_importance([param], "saliency")
    param.tensor.data[:] = param.data[::-1].copy()
    param.tensor.grad[:] = 100.0
    stats = apply_global_pruning_step([param], "saliency", 0.5, scores=scores)
    assert stats.active_params == 3
    np.testing.assert_array_equal(param.mask, [0, 0, 0, 0, 1, 1, 1])
    assert_pruning_budget([param], 0.5)
    assert active_budget(100, 0.9) == 10
    assert active_budget(234752, 0.95) == 11737


@pytest.mark.parametrize("start,end,epochs,interval", [(0, 2, 3, 1), (3, 2, 3, 1), (2, 4, 3, 1), (1, 2, 3, 0)])
def test_invalid_pruning_windows(start, end, epochs, interval):
    with pytest.raises(ValueError):
        pruning_events(10, start, end, epochs, interval)


def test_final_event_and_budget_check():
    assert pruning_events(3, 1, 1, 1, 100) == (1, 3)
    assert pruning_events(10, 1, 2, 3, 4) == (5, 20)
    with pytest.raises(RuntimeError):
        assert_pruning_budget([Parameter(np.ones(7), use_mask=True)], 0.5)


@pytest.mark.parametrize("script", ["train_self_pruning", "run_pruning_sweep"])
def test_trainers_capture_scores_before_update(monkeypatch, tmp_path, script):
    path = Path(__file__).resolve().parents[1] / "scripts" / (script + ".py")
    spec = importlib.util.spec_from_file_location(script, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(42)
    data = {"x_" + split: rng.normal(size=(3, 784)) for split in ["train", "val", "test"]}
    data.update({"y_" + split: np.array([0, 1, 2]) for split in ["train", "val", "test"]})
    monkeypatch.setattr(module, "load_mnist", lambda **kwargs: data)
    snapshots = []
    def capture(params, criterion):
        scores = capture_importance(params, criterion)
        snapshots.append({id(param): param.data.copy() for param in params})
        return scores
    def prune(params, criterion, target_sparsity, scores=None):
        assert scores is not None
        assert any(not np.array_equal(param.data, snapshots[-1][id(param)]) for param in params)
        for param in params:
            if param.mask is not None:
                np.testing.assert_allclose(scores[id(param)], np.abs(snapshots[-1][id(param)] * param.grad))
        return apply_global_pruning_step(params, criterion, target_sparsity, scores=scores)
    monkeypatch.setattr(module, "capture_importance", capture)
    monkeypatch.setattr(module, "apply_global_pruning_step", prune)
    if script == "run_pruning_sweep":
        run = module.train_once(7, "saliency", 0.5, 1, 2, 0.001, [3], 1, 1, 100)
        assert run["achieved_sparsity"] >= 0.5
    else:
        monkeypatch.setattr(sys, "argv", [script, "--epochs", "1", "--batch-size", "2", "--hidden", "3", "--prune-start-epoch", "1", "--prune-end-epoch", "1", "--prune-interval", "100", "--target-sparsity", "0.5", "--out-csv", str(tmp_path / "curve.csv")])
        module.main()
    assert len(snapshots) == 1


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
