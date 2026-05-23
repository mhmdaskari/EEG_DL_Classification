<p align="center">
  <a href="https://mhmdaskari.github.io/EEG_DL_Classification/">
    <img alt="Documentation" src="https://img.shields.io/badge/Documentation-Open%20Site-00897B?style=for-the-badge">
  </a>
</p>

# EEGClassify

`eegclassify` is a standalone Python package for reproducible EEG task classification. It provides a consistent data workflow, reusable preprocessing utilities, and CNN, LSTM, CNN-LSTM, and GAN-augmented training paths across TensorFlow, PyTorch, and JAX/Flax.

Use it when you want a clean package interface for running EEG classification experiments, comparing frameworks, or regenerating the processed BCI Competition IV-2a arrays from the public raw data.

## What Is Included

- Reproducible BCI Competition IV data set 2a download and conversion tools.
- Processed `.npy` data files tracked with Git LFS for quick local examples.
- Shared experiment settings for labels, splits, augmentation, seeds, and batching.
- TensorFlow/Keras, PyTorch, and JAX/Flax model implementations.
- Reusable GAN augmentation for classifier training data.
- Jupyter notebooks and CLI commands for smoke tests and full reproductions.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[data,notebooks,docs,dev]"
```

## Data Preparation

The public source of truth is BCI Competition IV, data set 2a.

- Dataset page: https://www.bbci.de/competition/iv/
- Raw GDF archive: https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
- True-label archive: https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip

Download and regenerate the processed arrays:

```bash
python -m eegclassify.cli download-bci2a --raw-dir data/raw
python -m eegclassify.cli prepare-bci2a \
  --raw-dir data/raw \
  --output-dir data/processed_regenerated
python -m eegclassify.cli compare-data \
  --generated-dir data/processed_regenerated \
  --reference-dir data/processed
```

## Notebook Examples

The runnable notebooks live under [`notebooks/`](notebooks/) and are also rendered in the documentation site.

- [Data postprocessing](notebooks/00_data_postprocessing.ipynb)
- [TensorFlow reproduction](notebooks/01_tensorflow_reproduction.ipynb)
- [PyTorch reproduction](notebooks/02_pytorch_reproduction.ipynb)
- [JAX/Flax reproduction](notebooks/03_jax_reproduction.ipynb)

Each notebook defaults to `FAST_DEV_RUN=True` so it can be opened and executed quickly before launching longer training runs.

## CLI Training

```bash
python -m eegclassify.cli train --framework tensorflow --model cnn --fast-dev-run
python -m eegclassify.cli train --framework pytorch --model cnn --fast-dev-run
python -m eegclassify.cli train --framework jax --model cnn --fast-dev-run
```

Use GAN augmentation with the canonical CNN path:

```bash
python -m eegclassify.cli train \
  --framework tensorflow \
  --model gan_cnn \
  --use-gan-augmentation
```

## GPU Extras

For NVIDIA GPU support on Linux, install the framework extras you need:

```bash
python -m pip install -e ".[tensorflow,pytorch,jax-cuda13]"
```

The TensorFlow CLI configures the pip-installed CUDA library paths before importing TensorFlow.

## Documentation And Package Notes

- Documentation: https://mhmdaskari.github.io/EEG_DL_Classification/
- Processed files under `data/processed/*.npy` are tracked with Git LFS.
- Processed arrays are excluded from PyPI builds; package users can download or regenerate them with the CLI.

Run checks locally:

```bash
pytest
mkdocs build --strict
```

This package was first initiated as part of a UCLA C247 course project; the original report is available as [Report.pdf](Report.pdf).

Archived notebook-only prototypes are kept under [`notebooks/legacy/`](notebooks/legacy/) for historical reference.
