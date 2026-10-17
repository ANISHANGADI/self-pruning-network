from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(seed: int = 7, val_size: int = 5000) -> dict[str, np.ndarray]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(np.int64)

    x_train_full = x[:60000]
    y_train_full = y[:60000]
    x_test = x[60000:]
    y_test = y[60000:]

    rng = np.random.default_rng(seed)
    idx = np.arange(x_train_full.shape[0])
    rng.shuffle(idx)

    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return {
        "x_train": x_train_full[train_idx],
        "y_train": y_train_full[train_idx],
        "x_val": x_train_full[val_idx],
        "y_val": y_train_full[val_idx],
        "x_test": x_test,
        "y_test": y_test,
    }
