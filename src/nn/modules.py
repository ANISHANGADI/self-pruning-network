from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from nn.parameter import Parameter


class Module:
    def parameters(self) -> list[Parameter]:
        return []

    def train(self) -> None:
        return

    def eval(self) -> None:
        return


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, use_mask: bool = True) -> None:
        # He initialization for ReLU-based networks.
        std = np.sqrt(2.0 / in_features)
        w = np.random.randn(in_features, out_features) * std
        b = np.zeros((1, out_features), dtype=np.float64)
        self.weight = Parameter(w, requires_grad=True, use_mask=use_mask)
        self.bias = Parameter(b, requires_grad=True, use_mask=False)

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.weight.masked()) + self.bias.tensor

    def parameters(self) -> list[Parameter]:
        return [self.weight, self.bias]


class ReLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def __call__(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MLP(Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, use_mask: bool = True) -> None:
        dims = [input_dim] + hidden_dims + [num_classes]
        layers: list[Module] = []
        for i in range(len(dims) - 2):
            layers.append(Linear(dims[i], dims[i + 1], use_mask=use_mask))
            layers.append(ReLU())
        layers.append(Linear(dims[-2], dims[-1], use_mask=use_mask))
        self.net = Sequential(*layers)

    def __call__(self, x: Tensor) -> Tensor:
        return self.net(x)

    def parameters(self) -> list[Parameter]:
        return self.net.parameters()
