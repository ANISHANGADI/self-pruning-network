import numpy as np
import pytest

from engine.tensor import Tensor, log_softmax, nll_loss


def numerical_grad_scalar(fn, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = fn(x)
        x[idx] = old - eps
        f_minus = fn(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def assert_close(a, b, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def test_add_gradcheck_broadcast():
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(4, 3))
    b_data = rng.normal(size=(3,))

    x = Tensor(x_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)
    out = (x + b).sum()
    out.backward()

    fn_x = lambda v: (v + b_data).sum()
    fn_b = lambda v: (x_data + v).sum()

    nx = numerical_grad_scalar(fn_x, x_data.copy())
    nb = numerical_grad_scalar(fn_b, b_data.copy())

    assert_close(x.grad, nx)
    assert_close(b.grad, nb)


def test_mul_gradcheck_broadcast():
    rng = np.random.default_rng(1)
    x_data = rng.normal(size=(5, 4))
    s_data = rng.normal(size=(4,))

    x = Tensor(x_data.copy(), requires_grad=True)
    s = Tensor(s_data.copy(), requires_grad=True)
    out = (x * s).sum()
    out.backward()

    fn_x = lambda v: (v * s_data).sum()
    fn_s = lambda v: (x_data * v).sum()

    nx = numerical_grad_scalar(fn_x, x_data.copy())
    ns = numerical_grad_scalar(fn_s, s_data.copy())

    assert_close(x.grad, nx)
    assert_close(s.grad, ns)


def test_matmul_gradcheck():
    rng = np.random.default_rng(2)
    a_data = rng.normal(size=(3, 4))
    b_data = rng.normal(size=(4, 2))

    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)
    out = (a @ b).sum()
    out.backward()

    fn_a = lambda v: (v @ b_data).sum()
    fn_b = lambda v: (a_data @ v).sum()

    na = numerical_grad_scalar(fn_a, a_data.copy())
    nb = numerical_grad_scalar(fn_b, b_data.copy())

    assert_close(a.grad, na)
    assert_close(b.grad, nb)


def test_relu_gradcheck():
    rng = np.random.default_rng(3)
    x_data = rng.normal(size=(10,))
    x_data[np.abs(x_data) < 0.1] += 0.2

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.relu().sum()
    out.backward()

    fn_x = lambda v: np.maximum(v, 0.0).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())

    assert_close(x.grad, nx)


def test_sigmoid_gradcheck():
    rng = np.random.default_rng(4)
    x_data = rng.normal(size=(6, 7))

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.sigmoid().sum()
    out.backward()

    fn_x = lambda v: (1.0 / (1.0 + np.exp(-v))).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())

    assert_close(x.grad, nx)


def test_logsoftmax_nll_backward_shape_and_finite():
    rng = np.random.default_rng(5)
    logits_data = rng.normal(size=(8, 10))
    targets = rng.integers(low=0, high=10, size=(8,))

    logits = Tensor(logits_data.copy(), requires_grad=True)
    log_probs = log_softmax(logits, axis=1)
    loss = nll_loss(log_probs, targets)
    loss.backward()

    assert logits.grad.shape == logits_data.shape
    assert np.isfinite(logits.grad).all()
    assert np.isfinite(loss.data).all()


def test_gradient_accumulation_shared_subgraph():
    rng = np.random.default_rng(6)
    x_data = rng.normal(size=(4, 4))

    x = Tensor(x_data.copy(), requires_grad=True)
    y = x * x
    z = y + y
    out = z.sum()
    out.backward()

    expected = 4.0 * x_data
    assert_close(x.grad, expected)


def test_exp_gradcheck():
    rng = np.random.default_rng(7)
    x_data = rng.normal(size=(3, 5))

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.exp().sum()
    out.backward()

    fn_x = lambda v: np.exp(v).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())
    assert_close(x.grad, nx)


def test_log_gradcheck():
    rng = np.random.default_rng(8)
    # Keep values safely positive for numerical stability around log.
    x_data = np.abs(rng.normal(size=(4, 4))) + 0.5

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.log().sum()
    out.backward()

    fn_x = lambda v: np.log(v).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())
    assert_close(x.grad, nx)


def test_pow_gradcheck():
    rng = np.random.default_rng(9)
    x_data = np.abs(rng.normal(size=(6,))) + 0.2
    exponent = 2.5

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.pow(exponent).sum()
    out.backward()

    fn_x = lambda v: (v**exponent).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())
    assert_close(x.grad, nx)


def test_reshape_gradcheck():
    rng = np.random.default_rng(10)
    x_data = rng.normal(size=(2, 3, 4))

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.reshape(4, 6).sum()
    out.backward()

    fn_x = lambda v: v.reshape(4, 6).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())
    assert_close(x.grad, nx)


def test_mean_axis_gradcheck():
    rng = np.random.default_rng(11)
    x_data = rng.normal(size=(3, 4, 5))

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.mean(axis=1).sum()
    out.backward()

    fn_x = lambda v: v.mean(axis=1).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())
    assert_close(x.grad, nx)


def test_sum_axis_keepdims_gradcheck():
    rng = np.random.default_rng(12)
    x_data = rng.normal(size=(3, 2, 4))

    x = Tensor(x_data.copy(), requires_grad=True)
    out = x.sum(axis=2, keepdims=True).sum()
    out.backward()

    fn_x = lambda v: v.sum(axis=2, keepdims=True).sum()
    nx = numerical_grad_scalar(fn_x, x_data.copy())
    assert_close(x.grad, nx)


@pytest.mark.parametrize("operation", [
    lambda value, other: value - other,
    lambda value, other: other - value,
    lambda value, other: value / other,
    lambda value, other: other / value,
])
def test_binary_arithmetic_gradcheck(operation):
    values = np.array([[0.8, 1.2, 2.0], [1.5, 0.7, 3.0]])
    other_values = np.array([1.1, 2.3, 0.9])
    upstream = np.array([[1.0, -2.0, 0.5], [0.3, 1.7, -0.8]])
    value = Tensor(values.copy(), requires_grad=True)
    other = Tensor(other_values.copy(), requires_grad=True)
    operation(value, other).backward(upstream)
    assert_close(value.grad, numerical_grad_scalar(lambda data: (operation(data, other_values) * upstream).sum(), values.copy()))
    assert_close(other.grad, numerical_grad_scalar(lambda data: (operation(values, data) * upstream).sum(), other_values.copy()))


@pytest.mark.parametrize("operation", [lambda value: -value, lambda value: 2.0 - value, lambda value: 2.0 / value, lambda value: 2.0 + value, lambda value: 2.0 * value])
def test_unary_and_reflected_arithmetic_gradcheck(operation):
    values = np.array([0.7, 1.5, 2.1])
    upstream = np.array([1.0, -2.0, 0.5])
    value = Tensor(values.copy(), requires_grad=True)
    operation(value).backward(upstream)
    assert_close(value.grad, numerical_grad_scalar(lambda data: (operation(data) * upstream).sum(), values.copy()))


def test_log_softmax_numerical_gradients():
    values = np.array([[1.2, -0.5, 0.3], [-0.7, 0.2, 1.4]])
    upstream = np.array([[0.3, -0.8, 1.1], [0.7, 0.4, -0.2]])
    value = Tensor(values.copy(), requires_grad=True)
    log_softmax(value).backward(upstream)
    def objective(data):
        shifted = data - data.max(axis=1, keepdims=True)
        return ((shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))) * upstream).sum()
    assert_close(value.grad, numerical_grad_scalar(objective, values.copy()))


def test_nll_and_cross_entropy_numerical_gradients():
    values = np.array([[1.2, -0.5, 0.3], [-0.7, 0.2, 1.4]])
    targets = np.array([2, 0])
    for combined in [False, True]:
        value = Tensor(values.copy(), requires_grad=True)
        nll_loss(log_softmax(value) if combined else value, targets).backward()
        def objective(data):
            if combined:
                shifted = data - data.max(axis=1, keepdims=True)
                data = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            return -data[np.arange(2), targets].mean()
        assert_close(value.grad, numerical_grad_scalar(objective, values.copy()))


@pytest.mark.parametrize("axis", [None, -1, (0, -1)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_reduction_variants_gradcheck(axis, keepdims, reduction):
    values = np.random.default_rng(23).normal(size=(2, 3, 4))
    value = Tensor(values.copy(), requires_grad=True)
    result = getattr(value, reduction)(axis=axis, keepdims=keepdims)
    upstream = np.random.default_rng(24).normal(size=result.shape)
    result.backward(upstream)
    assert_close(value.grad, numerical_grad_scalar(lambda data: (getattr(data, reduction)(axis=axis, keepdims=keepdims) * upstream).sum(), values.copy()))
