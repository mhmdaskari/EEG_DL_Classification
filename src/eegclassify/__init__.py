"""Tools for reproducible EEG classification experiments."""

from .config import (
    DataSplitConfig,
    DatasetPaths,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from .data import DatasetBundle, load_processed_arrays
from .preprocessing import PreparedSplits, prepare_splits

__all__ = [
    "DataSplitConfig",
    "DatasetBundle",
    "DatasetPaths",
    "ModelConfig",
    "PreparedSplits",
    "PreprocessingConfig",
    "TrainingConfig",
    "load_processed_arrays",
    "prepare_splits",
]

__version__ = "0.1.0"
