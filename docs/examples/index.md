# Notebook Examples

These notebooks provide guided, runnable examples for the main `eegclassify`
workflows. They are rendered in the documentation and also kept as source notebooks
under the repository's `notebooks/` directory.

Each training notebook starts with `FAST_DEV_RUN=True`. That setting keeps examples
short enough for a first pass while preserving the same configuration cells used for
longer runs.

## Available Examples

- [Data postprocessing](00_data_postprocessing.ipynb): download, convert, summarize, and compare processed EEG arrays.
- [TensorFlow reproduction](01_tensorflow_reproduction.ipynb): train and evaluate TensorFlow/Keras classifiers.
- [PyTorch reproduction](02_pytorch_reproduction.ipynb): train and evaluate PyTorch classifiers.
- [JAX/Flax reproduction](03_jax_reproduction.ipynb): train and evaluate JAX/Flax classifiers.

For scripted runs, use the commands in [Reproduce Results](../reproduce.md).
