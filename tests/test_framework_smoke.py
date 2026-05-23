from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from eegclassify.config import ModelConfig, TrainingConfig


def test_tensorflow_cnn_forward_if_installed():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed.")
    from eegclassify.models.tensorflow import TensorFlowClassifierFactory

    config = ModelConfig(max_time_step=20)
    model = TensorFlowClassifierFactory(config).build_cnn()
    y = model.predict(np.zeros((2, 10, 1, 22), dtype=np.float32), verbose=0)
    assert y.shape == (2, 4)


def test_pytorch_cnn_forward_if_installed():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed.")
    import torch
    from eegclassify.models.pytorch import PyTorchCNN

    model = PyTorchCNN(ModelConfig(max_time_step=20))
    y = model(torch.zeros(2, 10, 1, 22))
    assert tuple(y.shape) == (2, 4)


def test_jax_cnn_forward_if_installed():
    if importlib.util.find_spec("jax") is None or importlib.util.find_spec("flax") is None:
        pytest.skip("JAX/Flax is not installed.")
    import jax
    import jax.numpy as jnp
    from eegclassify.models.jax import FlaxCNN

    model = FlaxCNN(ModelConfig(max_time_step=20))
    variables = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, jnp.zeros((2, 10, 1, 22)), train=False)
    y = model.apply(variables, jnp.zeros((2, 10, 1, 22)), train=False)
    assert tuple(y.shape) == (2, 4)
