from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from eegclassify.augmentation import (
    append_synthetic_training_data,
    classifier_images_to_gan_sequences,
    interpolation_labels,
)
from eegclassify.config import ModelConfig, TrainingConfig


def _skip_heavy_framework_gan_tests() -> None:
    if os.environ.get("EEGCLASSIFY_RUN_FRAMEWORK_GAN_TESTS") != "1":
        pytest.skip("Set EEGCLASSIFY_RUN_FRAMEWORK_GAN_TESTS=1 to run framework GAN training smoke tests.")


def test_interpolation_labels_follow_legacy_ordered_class_pairs():
    labels = interpolation_labels(n_classes=4, samples_per_class=3)

    assert labels.shape == (48, 4)
    np.testing.assert_allclose(labels[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(labels[2], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(labels[3], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(labels[5], [0.0, 1.0, 0.0, 0.0])


def test_gan_append_only_changes_training_arrays():
    x_train = np.ones((2, 250, 1, 22), dtype=np.float32)
    y_train = np.eye(4, dtype=np.float32)[:2]
    synthetic = np.full((3, 250, 22), 0.75, dtype=np.float32)
    synthetic_labels = np.eye(4, dtype=np.float32)[:3]

    result = append_synthetic_training_data(x_train, y_train, synthetic, synthetic_labels, scale=2.0)

    assert result.x_train.shape == (5, 250, 1, 22)
    assert result.y_train.shape == (5, 4)
    np.testing.assert_allclose(result.x_train[:2], x_train)
    np.testing.assert_allclose(result.y_train[:2], y_train)
    np.testing.assert_allclose(result.x_synthetic, np.ones((3, 250, 1, 22), dtype=np.float32))


def test_gan_normalization_handles_zero_signal():
    x_train = np.zeros((2, 250, 1, 22), dtype=np.float32)

    normalized, scale = classifier_images_to_gan_sequences(x_train)

    assert scale == 1.0
    assert normalized.shape == (2, 250, 22)
    np.testing.assert_allclose(normalized, 0.5)


def test_tensorflow_gan_augmentation_if_installed():
    _skip_heavy_framework_gan_tests()
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed.")
    from eegclassify.models.tensorflow import TensorFlowGANAugmenter

    x_train = np.random.default_rng(1).normal(size=(4, 250, 1, 22)).astype(np.float32)
    y_train = np.eye(4, dtype=np.float32)
    training = TrainingConfig(fast_dev_run=True, gan_batch_size=2, gan_samples_per_class=1)
    result = TensorFlowGANAugmenter(training, ModelConfig()).augment_training_data(x_train, y_train)

    assert result.x_train.shape[0] == 20
    assert result.x_synthetic.shape == (16, 250, 1, 22)


def test_pytorch_gan_augmentation_if_installed():
    _skip_heavy_framework_gan_tests()
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed.")
    from eegclassify.models.pytorch import PyTorchGANAugmenter

    x_train = np.random.default_rng(1).normal(size=(4, 250, 1, 22)).astype(np.float32)
    y_train = np.eye(4, dtype=np.float32)
    training = TrainingConfig(fast_dev_run=True, gan_batch_size=2, gan_samples_per_class=1)
    result = PyTorchGANAugmenter(training, ModelConfig(), device="cpu").augment_training_data(x_train, y_train)

    assert result.x_train.shape[0] == 20
    assert result.x_synthetic.shape == (16, 250, 1, 22)


def test_jax_gan_augmentation_if_installed():
    _skip_heavy_framework_gan_tests()
    if importlib.util.find_spec("jax") is None or importlib.util.find_spec("flax") is None:
        pytest.skip("JAX/Flax is not installed.")
    from eegclassify.models.jax import JAXGANAugmenter

    x_train = np.random.default_rng(1).normal(size=(4, 250, 1, 22)).astype(np.float32)
    y_train = np.eye(4, dtype=np.float32)
    training = TrainingConfig(fast_dev_run=True, gan_batch_size=2, gan_samples_per_class=1)
    result = JAXGANAugmenter(training, ModelConfig()).augment_training_data(x_train, y_train)

    assert result.x_train.shape[0] == 20
    assert result.x_synthetic.shape == (16, 250, 1, 22)
