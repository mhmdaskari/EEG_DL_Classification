"""Dataclass configuration objects used across the package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem locations used by data and experiment workflows.

    Attributes:
        raw_dir: Directory for official downloaded BCI IV-2a archives and extracted GDF files.
        processed_dir: Directory for generated project `.npy` arrays.
        local_cache_dir: Optional directory containing an existing local cache of the six `.npy` files.
        artifacts_dir: Directory for generated reports, metrics, and plots.
    """

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    local_cache_dir: Path | None = Path("data_temp")
    artifacts_dir: Path = Path("artifacts/runs")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing and augmentation settings.

    Attributes:
        sub_sample: Temporal stride used for max-pooling and subsampling augmentation.
        average: Window size used for average-pooling augmentation.
        noise: Whether to add Gaussian noise to averaged/subsampled augmentations.
        noise_std: Standard deviation of Gaussian augmentation noise.
        max_time_step: Number of raw time samples kept before temporal reduction.
        n_classes: Number of output classes.
        label_offset: Offset used to convert BCI cue labels `769-772` to `0-3`.
        seed: Random seed for deterministic validation splits and augmentation.
    """

    sub_sample: int = 2
    average: int = 2
    noise: bool = True
    noise_std: float = 0.5
    max_time_step: int = 500
    n_classes: int = 4
    label_offset: int = 769
    seed: int = 1


@dataclass(frozen=True)
class DataSplitConfig:
    """Train/validation/test split settings.

    Attributes:
        validation_ratio: Fraction of train/valid trials assigned to validation.
        subject_id: Optional subject ID `0-8`; `None` uses all subjects.
        seed: Random seed used for validation-index sampling.
    """

    validation_ratio: float = 0.17
    subject_id: int | None = None
    seed: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Training defaults shared by examples and notebook runs.

    Attributes:
        epochs: Number of classifier training epochs for full reproduction.
        gan_epochs: Number of conditional GAN training epochs.
        batch_size: Classifier batch size.
        gan_batch_size: Conditional GAN batch size.
        learning_rate: Classifier Adam learning rate.
        gan_learning_rate: Conditional GAN Adam learning rate.
        use_gan_augmentation: Whether to append synthetic GAN samples to the training split.
        gan_samples_per_class: Number of interpolation samples generated for each ordered
            class pair in the GAN augmentation routine.
        seed: Framework and data random seed.
        fast_dev_run: If `True`, notebooks/trainers run a one-epoch smoke check.
    """

    epochs: int = 100
    gan_epochs: int = 20
    batch_size: int = 64
    gan_batch_size: int = 128
    learning_rate: float = 1e-3
    gan_learning_rate: float = 3e-4
    use_gan_augmentation: bool = False
    gan_samples_per_class: int = 10
    seed: int = 1
    fast_dev_run: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture defaults.

    Attributes:
        channels: Number of EEG channels.
        n_classes: Number of classification targets.
        max_time_step: Raw time samples used before preprocessing reduction.
        dropout: Dropout probability used in CNN blocks.
        cnn_filters: Filters for the four convolutional blocks.
        cnn_kernel: Convolution kernel size for CNN-style classifiers.
        lstm_units: Hidden sizes for the stacked bidirectional LSTM classifier.
        cnn_lstm_units: Hidden size for the CNN-LSTM recurrent layer.
        latent_dim: Latent noise dimension used by conditional GAN components.
    """

    channels: int = 22
    n_classes: int = 4
    max_time_step: int = 500
    dropout: float = 0.5
    cnn_filters: tuple[int, int, int, int] = (25, 50, 100, 200)
    cnn_kernel: tuple[int, int] = (10, 1)
    lstm_units: tuple[int, int, int] = (32, 16, 8)
    cnn_lstm_units: int = 10
    latent_dim: int = 128
