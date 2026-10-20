import numpy as np
import pytest

from engine.tensor import Tensor
from nn.modules import MLP, Linear
from prune.cost import dense_forward_numpy, sparse_forward_numpy, summarize_cost


@pytest.mark.parametrize("sparsity", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("use_mask", [False, True])
def test_forward_equivalence_and_counts(sparsity, batch_size, use_mask):
    np.random.seed(42)
    rng = np.random.default_rng(7)
    model = MLP(4, [3], 2, use_mask=use_mask)
    for layer in model.net.layers:
        if isinstance(layer, Linear):
            layer.bias.tensor.data[:] = rng.normal(size=layer.bias.data.shape)
            if use_mask:
                layer.weight.set_mask((rng.random(layer.weight.data.shape) >= sparsity).astype(float))
    inputs = rng.normal(size=(batch_size, 4))
    reference = model(Tensor(inputs)).data
    np.testing.assert_allclose(dense_forward_numpy(model, inputs), reference, atol=1e-12)
    np.testing.assert_allclose(sparse_forward_numpy(model, inputs), reference, atol=1e-12)
    cost = summarize_cost(model, inputs, repeats=1)
    assert cost.dense_params == cost.dense_macs_per_example == 18
    assert cost.active_params == cost.sparse_macs_per_example == sum(layer.weight.num_active() for layer in model.net.layers if isinstance(layer, Linear))
    assert np.isfinite([cost.dense_ms_per_batch, cost.sparse_ms_per_batch]).all()


def test_unsupported_inference_layer():
    model = MLP(2, [], 2)
    model.net.layers.append(object())
    for forward in [dense_forward_numpy, sparse_forward_numpy]:
        with pytest.raises(TypeError):
            forward(model, np.ones((1, 2)))