from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class Parameter:
    def __init__(self, data: np.ndarray, requires_grad: bool = True, use_mask: bool = False) -> None:
        self.tensor = Tensor(data.astype(np.float64), requires_grad=requires_grad)
        self.mask: np.ndarray | None = None
        if use_mask:
            self.mask = np.ones_like(self.tensor.data, dtype=np.float64)

    @property
    def data(self) -> np.ndarray:
        return self.tensor.data

    @property
    def grad(self) -> np.ndarray | None:
        return self.tensor.grad

    def zero_grad(self) -> None:
        self.tensor.zero_grad()

    def masked(self) -> Tensor:
        if self.mask is None:
            return self.tensor
        return self.tensor * Tensor(self.mask)

    def set_mask(self, new_mask: np.ndarray) -> None:
        if self.mask is None:
            self.mask = np.ones_like(self.tensor.data, dtype=np.float64)
        if new_mask.shape != self.tensor.data.shape:
            raise ValueError("new_mask shape must match parameter shape")
        self.mask = new_mask.astype(np.float64)
        self.tensor.data *= self.mask
        if self.tensor.grad is not None:
            self.tensor.grad *= self.mask

    def apply_mask_inplace(self) -> None:
        if self.mask is None:
            return
        self.tensor.data *= self.mask
        if self.tensor.grad is not None:
            self.tensor.grad *= self.mask

    def num_active(self) -> int:
        if self.mask is None:
            return int(np.prod(self.tensor.data.shape))
        return int(np.count_nonzero(self.mask))

    def num_total(self) -> int:
        return int(np.prod(self.tensor.data.shape))
