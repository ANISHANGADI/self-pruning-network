from __future__ import annotations

import numpy as np

from nn.parameter import Parameter


class Adam:
    def __init__(
        self,
        params: list[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.state: dict[int, dict[str, np.ndarray]] = {}

        for p in self.params:
            pid = id(p)
            self.state[pid] = {
                "m": np.zeros_like(p.data),
                "v": np.zeros_like(p.data),
                "prev_mask": None if p.mask is None else p.mask.copy(),
            }

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            pid = id(p)
            st = self.state[pid]
            m = st["m"]
            v = st["v"]

            g = p.grad.copy()
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            if p.mask is not None:
                # Masked entries should not receive gradient updates.
                g *= p.mask

                prev_mask = st["prev_mask"]
                if prev_mask is not None:
                    revived = (p.mask > 0.0) & (prev_mask == 0.0)
                    if np.any(revived):
                        # Prevent stale moments from pre-pruning history.
                        m[revived] = 0.0
                        v[revived] = 0.0
                st["prev_mask"] = p.mask.copy()

            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            p.tensor.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if p.mask is not None:
                # Enforce hard zeros post-step and keep moments clean on masked coords.
                p.tensor.data *= p.mask
                if p.tensor.grad is not None:
                    p.tensor.grad *= p.mask
                m *= p.mask
                v *= p.mask
