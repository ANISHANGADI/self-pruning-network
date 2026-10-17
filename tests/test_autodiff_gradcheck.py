import numpy as np

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
