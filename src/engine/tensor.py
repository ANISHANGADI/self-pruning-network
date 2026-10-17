from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np


BackwardFn = Callable[[np.ndarray], None]


def _to_numpy(data: np.ndarray | float | int) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float64, copy=False)
    return np.array(data, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    # Sum out leading dimensions that came from broadcasting.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum axes where target had size 1.
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


@dataclass(eq=False)
class Tensor:
    data: np.ndarray | float | int
    requires_grad: bool = False
    _children: tuple["Tensor", ...] = field(default_factory=tuple)
    _op: str = ""

    def __post_init__(self) -> None:
        self.data = _to_numpy(self.data)
        self.grad: np.ndarray | None = np.zeros_like(self.data) if self.requires_grad else None
        self._backward: BackwardFn = lambda _g: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    def backward(self, grad: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot backprop through a tensor that does not require gradients.")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("Non-scalar tensors require an explicit initial gradient.")
            grad = np.ones_like(self.data)
        else:
            grad = _to_numpy(grad)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(v: Tensor) -> None:
            node_id = id(v)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent in v._children:
                build(parent)
            topo.append(v)

        build(self)

        self.grad = grad.copy()
        for node in reversed(topo):
            if node.grad is None:
                continue
            node._backward(node.grad)

    @staticmethod
    def _ensure_tensor(other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="add")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(grad, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self + other

    def __sub__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self + (-Tensor._ensure_tensor(other))

    def __rsub__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return Tensor._ensure_tensor(other) - self

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(-grad)

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="mul")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return self * other

    def __truediv__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        other = Tensor._ensure_tensor(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other: "Tensor" | np.ndarray | float | int) -> "Tensor":
        return Tensor._ensure_tensor(other) / self

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires another Tensor")
        with np.errstate(all="ignore"):
            out_data = self.data @ other.data
        if not np.isfinite(out_data).all():
            raise FloatingPointError("Non-finite values produced in matmul forward pass")
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="matmul")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                with np.errstate(all="ignore"):
                    grad_self = grad @ other.data.T
                if not np.isfinite(grad_self).all():
                    raise FloatingPointError("Non-finite gradient for matmul lhs")
                self._accumulate_grad(grad_self)
            if other.requires_grad:
                with np.errstate(all="ignore"):
                    grad_other = self.data.T @ grad
                if not np.isfinite(grad_other).all():
                    raise FloatingPointError("Non-finite gradient for matmul rhs")
                other._accumulate_grad(grad_other)

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward(grad: np.ndarray) -> None:
            if not self.requires_grad:
                return
            g = grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                normalized_axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                for ax in sorted(normalized_axes):
                    g = np.expand_dims(g, axis=ax)
            self._accumulate_grad(np.ones_like(self.data) * g)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            denom = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="exp")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(grad * out_data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op="log")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(grad / self.data)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out_data = np.maximum(self.data, 0.0)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        out_data = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(grad * out_data * (1.0 - out_data))

        out._backward = _backward
        return out

    def pow(self, exponent: float) -> "Tensor":
        out_data = self.data ** exponent
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="pow")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(grad * exponent * (self.data ** (exponent - 1.0)))

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op="reshape")

        def _backward(grad: np.ndarray) -> None:
            if self.requires_grad:
                self._accumulate_grad(grad.reshape(self.shape))

        out._backward = _backward
        return out


def log_softmax(x: Tensor, axis: int = 1) -> Tensor:
    # Stable log-sum-exp trick.
    max_vals = Tensor(np.max(x.data, axis=axis, keepdims=True))
    shifted = x - max_vals
    lse = shifted.exp().sum(axis=axis, keepdims=True).log()
    return shifted - lse


def nll_loss(log_probs: Tensor, targets: np.ndarray) -> Tensor:
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D class-index array")
    batch = log_probs.data.shape[0]
    gathered = log_probs.data[np.arange(batch), targets]
    out = Tensor(-gathered.mean(), requires_grad=log_probs.requires_grad, _children=(log_probs,), _op="nll")

    def _backward(grad: np.ndarray) -> None:
        if not log_probs.requires_grad:
            return
        g = np.zeros_like(log_probs.data)
        g[np.arange(batch), targets] = -1.0 / batch
        log_probs._accumulate_grad(g * grad)

    out._backward = _backward
    return out
