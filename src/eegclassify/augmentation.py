"""Shared utilities for GAN-based training-data augmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GANAugmentationResult:
    """Output from a framework-specific GAN augmentation pass.

    Attributes:
        x_train: Training inputs after appending synthetic samples.
        y_train: Training labels after appending synthetic labels.
        x_synthetic: Synthetic classifier inputs shaped `(n, time, 1, channels)`.
        y_synthetic: Synthetic soft labels shaped `(n, classes)`.
        history: Per-epoch GAN losses or framework-specific training history.
        scale: Scale factor used to denormalize generated GAN sequences.
        metadata: Extra JSON-serializable details about the augmentation run.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_synthetic: np.ndarray
    y_synthetic: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


def interpolation_labels(n_classes: int = 4, samples_per_class: int = 10) -> np.ndarray:
    """Create GAN interpolation labels for all ordered class pairs.

    The routine generates `samples_per_class` points between every ordered pair of
    class labels, including same-class pairs. The generated labels are soft labels and
    can be used directly with categorical cross-entropy style losses.

    Args:
        n_classes: Number of class labels.
        samples_per_class: Interpolation points generated for each ordered class pair.

    Returns:
        Float32 label matrix shaped `(n_classes * n_classes * samples_per_class, n_classes)`.

    Raises:
        ValueError: If `n_classes` or `samples_per_class` is not positive.
    """

    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    labels: list[np.ndarray] = []
    weights = np.linspace(0.0, 1.0, samples_per_class, dtype=np.float32)[:, None]
    eye = np.eye(n_classes, dtype=np.float32)
    for start_class in range(n_classes):
        for end_class in range(n_classes):
            labels.append(eye[start_class] * (1.0 - weights) + eye[end_class] * weights)
    return np.vstack(labels).astype(np.float32)


def classifier_images_to_gan_sequences(x_train: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize classifier inputs into the sequence format expected by GANs.

    Args:
        x_train: Classifier inputs shaped `(n, time, 1, channels)`.

    Returns:
        Tuple of normalized sequences shaped `(n, time, channels)` and the scale factor
        required to map generated values back to the classifier scale.

    Raises:
        ValueError: If the input does not use the classifier image layout.
    """

    x_train = np.asarray(x_train)
    if x_train.ndim != 4 or x_train.shape[2] != 1:
        raise ValueError(f"Expected classifier inputs shaped (n, time, 1, channels), got {x_train.shape}.")
    sequences = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[3])
    scale = float(np.max(np.abs(sequences)))
    if scale == 0.0:
        scale = 1.0
    normalized = (sequences / (2.0 * scale) + 0.5).astype(np.float32)
    return normalized, scale


def gan_sequences_to_classifier_images(sequences: np.ndarray, scale: float) -> np.ndarray:
    """Denormalize generated GAN sequences back to classifier input layout.

    Args:
        sequences: GAN outputs shaped `(n, time, channels)` in the normalized `[0, 1]` range.
        scale: Scale factor returned by `classifier_images_to_gan_sequences`.

    Returns:
        Float32 classifier inputs shaped `(n, time, 1, channels)`.

    Raises:
        ValueError: If `sequences` is not three-dimensional.
    """

    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError(f"Expected generated sequences shaped (n, time, channels), got {sequences.shape}.")
    denormalized = (sequences - 0.5) * 2.0 * float(scale)
    return denormalized[:, :, None, :].astype(np.float32)


def append_synthetic_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    synthetic_sequences: np.ndarray,
    synthetic_labels: np.ndarray,
    scale: float,
    history: list[dict[str, float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GANAugmentationResult:
    """Append generated GAN samples to classifier training arrays.

    Args:
        x_train: Real classifier inputs shaped `(n, time, 1, channels)`.
        y_train: Real one-hot training labels.
        synthetic_sequences: Generated GAN samples shaped `(m, time, channels)`.
        synthetic_labels: Soft labels aligned with `synthetic_sequences`.
        scale: Scale factor used to denormalize generated sequences.
        history: Optional GAN training history.
        metadata: Optional run metadata.

    Returns:
        `GANAugmentationResult` with real and synthetic samples concatenated.
    """

    x_synthetic = gan_sequences_to_classifier_images(synthetic_sequences, scale)
    y_synthetic = np.asarray(synthetic_labels, dtype=np.float32)
    return GANAugmentationResult(
        x_train=np.concatenate([np.asarray(x_train, dtype=np.float32), x_synthetic], axis=0),
        y_train=np.concatenate([np.asarray(y_train, dtype=np.float32), y_synthetic], axis=0),
        x_synthetic=x_synthetic,
        y_synthetic=y_synthetic,
        history=history or [],
        scale=scale,
        metadata=metadata or {},
    )
