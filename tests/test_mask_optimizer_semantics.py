import numpy as np

from nn.parameter import Parameter
from optim.adam import Adam


def test_masked_coordinates_stay_zero_and_unupdated():
    p = Parameter(np.array([1.0, -2.0, 3.0]), use_mask=True)
    p.set_mask(np.array([1.0, 0.0, 1.0]))

    opt = Adam([p], lr=0.1)

    p.tensor.grad = np.array([0.5, 10.0, -0.5], dtype=np.float64)
    opt.step()

    assert p.data[1] == 0.0
    st = opt.state[id(p)]
    assert st["m"][1] == 0.0
    assert st["v"][1] == 0.0


def test_revived_coordinate_resets_stale_moments():
    p = Parameter(np.array([1.0]), use_mask=True)
    opt = Adam([p], lr=0.1)

    p.tensor.grad = np.array([1.0], dtype=np.float64)
    opt.step()
    st = opt.state[id(p)]
    assert st["m"][0] != 0.0
    assert st["v"][0] != 0.0

    p.set_mask(np.array([0.0]))
    p.tensor.grad = np.array([5.0], dtype=np.float64)
    opt.step()
    assert p.data[0] == 0.0
    assert st["m"][0] == 0.0
    assert st["v"][0] == 0.0

    # Manually inject stale state to simulate a buggy implementation.
    st["m"][0] = 123.0
    st["v"][0] = 456.0

    p.set_mask(np.array([1.0]))
    p.tensor.grad = np.array([0.2], dtype=np.float64)
    opt.step()

    # If revive-reset is correct, moments should be small and tied to new grad,
    # not dominated by stale historical values.
    assert abs(st["m"][0]) < 1.0
    assert abs(st["v"][0]) < 1.0


def test_masked_forward_blocks_gradient_flow_on_pruned_weights():
    w = Parameter(np.array([[1.0, -2.0], [3.0, 4.0]]), use_mask=True)
    w.set_mask(np.array([[1.0, 0.0], [0.0, 1.0]]))

    x = np.array([[2.0, -1.0]])
    from engine.tensor import Tensor

    xt = Tensor(x)
    out = xt @ w.masked()
    loss = out.sum()
    loss.backward()

    grad = w.grad
    assert grad is not None
    assert grad[0, 1] == 0.0
    assert grad[1, 0] == 0.0
    assert grad[0, 0] != 0.0
    assert grad[1, 1] != 0.0
