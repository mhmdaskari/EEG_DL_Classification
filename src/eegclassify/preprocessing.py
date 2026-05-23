"""Preprocessing, augmentation, and split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .augmentation import classifier_images_to_gan_sequences
from .config import DataSplitConfig, PreprocessingConfig
from .data import DatasetBundle


@dataclass
class PreparedSplits:
    """Preprocessed arrays ready for classifier training.

    Attributes:
        x_train: Augmented training inputs shaped `(n, time, 1, channels)`.
        y_train: One-hot training labels shaped `(n, 4)`.
        x_valid: Augmented validation inputs.
        y_valid: One-hot validation labels.
        x_test: Augmented held-out test inputs.
        y_test: One-hot held-out test labels.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def set_numpy_seed(seed: int) -> None:
    """Set NumPy's global random seed.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)


def labels_to_zero_based(y: np.ndarray, label_offset: int = 769) -> np.ndarray:
    """Convert BCI IV-2a event labels `769-772` to class IDs `0-3`.

    Args:
        y: Raw BCI cue labels.
        label_offset: Value subtracted from each label.

    Returns:
        Integer class labels in `[0, 3]`.

    Raises:
        ValueError: If converted labels fall outside `[0, 3]`.
    """

    y0 = np.asarray(y).astype(np.int64) - label_offset
    if np.any((y0 < 0) | (y0 > 3)):
        raise ValueError("Expected BCI cue labels 769-772 before zero-based conversion.")
    return y0


def one_hot(y: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Convert integer class labels to one-hot rows.

    Args:
        y: Integer labels.
        n_classes: Number of classes.

    Returns:
        Float32 one-hot array shaped `(len(y), n_classes)`.
    """

    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    result = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def augment_trials(
    X: np.ndarray,
    y: np.ndarray,
    config: PreprocessingConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the original notebook augmentation routine.

    The routine trims each trial, creates a max-pooled copy, creates an averaged copy
    with Gaussian noise, and appends each temporal subsampling offset.

    Args:
        X: EEG trials shaped `(trials, channels, time)`.
        y: Trial labels aligned with `X`.
        config: Preprocessing and augmentation settings.
        rng: Optional NumPy random generator.

    Returns:
        Tuple of augmented trials and aligned labels.

    Raises:
        ValueError: If array dimensions or temporal settings are incompatible.
    """

    if rng is None:
        rng = np.random.default_rng(config.seed)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (trials, channels, time), got {X.shape}.")
    if config.max_time_step <= 0:
        raise ValueError("max_time_step must be positive.")
    if config.max_time_step > X.shape[2]:
        raise ValueError("max_time_step cannot exceed the input time dimension.")
    if config.max_time_step % config.sub_sample != 0:
        raise ValueError("max_time_step must be divisible by sub_sample.")
    if config.max_time_step % config.average != 0:
        raise ValueError("max_time_step must be divisible by average.")

    trimmed = X[:, :, : config.max_time_step]
    X_max = np.max(trimmed.reshape(trimmed.shape[0], trimmed.shape[1], -1, config.sub_sample), axis=3)
    total_X = X_max
    total_y = y

    X_average = np.mean(trimmed.reshape(trimmed.shape[0], trimmed.shape[1], -1, config.average), axis=3)
    X_average = X_average + rng.normal(0.0, config.noise_std, X_average.shape)
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))

    for i in range(config.sub_sample):
        noise = rng.normal(0.0, config.noise_std, trimmed[:, :, i:: config.sub_sample].shape)
        X_subsample = trimmed[:, :, i:: config.sub_sample] + (noise if config.noise else 0.0)
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
    return total_X, total_y


def to_channels_last_image(X: np.ndarray) -> np.ndarray:
    """Convert `(trials, channels, time)` to `(trials, time, 1, channels)`.

    Args:
        X: EEG trials shaped `(trials, channels, time)`.

    Returns:
        Float32 channels-last image-like representation used by classifiers.
    """

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = np.swapaxes(X, 1, 3)
    X = np.swapaxes(X, 1, 2)
    return X.astype(np.float32)


def select_subject(
    X: np.ndarray,
    y: np.ndarray,
    persons: np.ndarray,
    subject_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter arrays to one subject, or return copies for all subjects.

    Args:
        X: EEG trial array.
        y: Labels aligned with `X`.
        persons: Subject IDs aligned with `X`.
        subject_id: Subject ID to keep, or `None` for all subjects.

    Returns:
        Filtered `(X, y)` pair.

    Raises:
        ValueError: If `subject_id` is provided but no matching trials exist.
    """

    if subject_id is None:
        return np.copy(X), np.copy(y)
    persons_flat = np.asarray(persons).reshape(-1)
    mask = persons_flat == subject_id
    if not np.any(mask):
        raise ValueError(f"No trials found for subject_id={subject_id}.")
    return X[mask], y[mask]


def prepare_splits(
    bundle: DatasetBundle,
    preprocess: PreprocessingConfig | None = None,
    split: DataSplitConfig | None = None,
) -> PreparedSplits:
    """Create augmented train/validation/test arrays ready for classifiers.

    Args:
        bundle: Raw processed arrays in the original six-file layout.
        preprocess: Optional preprocessing configuration.
        split: Optional split configuration.

    Returns:
        `PreparedSplits` containing channels-last inputs and one-hot labels.
    """

    preprocess = preprocess or PreprocessingConfig()
    split = split or DataSplitConfig(seed=preprocess.seed)
    rng = np.random.default_rng(split.seed)

    y_train_valid = labels_to_zero_based(bundle.y_train_valid, preprocess.label_offset)
    y_test = labels_to_zero_based(bundle.y_test, preprocess.label_offset)
    X_train_sub, y_train_sub = select_subject(
        bundle.X_train_valid, y_train_valid, bundle.person_train_valid, split.subject_id
    )
    X_test_sub, y_test_sub = select_subject(bundle.X_test, y_test, bundle.person_test, split.subject_id)

    n_train_valid = X_train_sub.shape[0]
    n_valid = int(np.floor(split.validation_ratio * n_train_valid))
    valid_idx = rng.choice(n_train_valid, n_valid, replace=False)
    train_idx = np.array(sorted(set(range(n_train_valid)).difference(set(valid_idx))))

    X_train, X_valid = X_train_sub[train_idx], X_train_sub[valid_idx]
    y_train, y_valid = y_train_sub[train_idx], y_train_sub[valid_idx]

    X_train_aug, y_train_aug = augment_trials(X_train, y_train, preprocess, rng)
    X_valid_aug, y_valid_aug = augment_trials(X_valid, y_valid, preprocess, rng)
    X_test_aug, y_test_aug = augment_trials(X_test_sub, y_test_sub, preprocess, rng)

    return PreparedSplits(
        x_train=to_channels_last_image(X_train_aug),
        y_train=one_hot(y_train_aug, preprocess.n_classes),
        x_valid=to_channels_last_image(X_valid_aug),
        y_valid=one_hot(y_valid_aug, preprocess.n_classes),
        x_test=to_channels_last_image(X_test_aug),
        y_test=one_hot(y_test_aug, preprocess.n_classes),
    )


def prepare_gan_training_arrays(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize classifier arrays into the GAN format used by the original notebook.

    Args:
        x_train: Classifier inputs shaped `(n, time, 1, channels)`.
        y_train: One-hot labels.

    Returns:
        Tuple of normalized sequence data `(n, time, channels)`, labels, and the
        scale factor needed to invert the normalization.
    """

    normalized, scale = classifier_images_to_gan_sequences(x_train)
    return normalized, y_train.astype(np.float32), scale
