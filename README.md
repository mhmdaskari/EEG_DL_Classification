<p align="center">
  <a href="https://mhmdaskari.github.io/EEG_DL_Classification/">
    <img alt="Documentation" src="https://img.shields.io/badge/Documentation-Open%20Site-00897B?style=for-the-badge">
  </a>
</p>

# EEGClassify

This repository contains `eegclassify`, a reproducible EEG classification package built from the original C247 project notebooks. It includes CNN, LSTM, CNN-LSTM, and GAN-augmented classifier experiments across TensorFlow, PyTorch, and JAX/Flax.

The original project write-up is available as [Report.pdf](Report.pdf). The original notebook code is preserved under [`notebooks/legacy/`](notebooks/legacy/) for reference, while the reproducible package-oriented notebooks live directly under [`notebooks/`](notebooks/).

The public data source is BCI Competition IV, data set 2a:

- Dataset page: https://www.bbci.de/competition/iv/
- Raw GDF archive: https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
- True-label archive: https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[data,notebooks,docs,dev]"
```

Download and prepare data:

```bash
python -m eegclassify.cli download-bci2a --raw-dir data/raw
python -m eegclassify.cli prepare-bci2a --raw-dir data/raw --output-dir data/processed
python -m eegclassify.cli compare-data --generated-dir data/processed --reference-dir data_temp
```

Run tests and build documentation:

```bash
pytest
mkdocs build --strict
```

## Notebook Examples

The main reproducible Jupyter notebook examples are:

- [Data postprocessing](notebooks/00_data_postprocessing.ipynb)
- [TensorFlow reproduction](notebooks/01_tensorflow_reproduction.ipynb)
- [PyTorch reproduction](notebooks/02_pytorch_reproduction.ipynb)
- [JAX/Flax reproduction](notebooks/03_jax_reproduction.ipynb)

The original project notebooks are preserved in [notebooks/legacy](notebooks/legacy/).

Run a quick local training smoke test:

```bash
python -m eegclassify.cli train --framework tensorflow --model cnn --fast-dev-run
python -m eegclassify.cli train --framework pytorch --model cnn --fast-dev-run
python -m eegclassify.cli train --framework jax --model cnn --fast-dev-run
```

Use the original GAN-CNN parity path with:

```bash
python -m eegclassify.cli train --framework tensorflow --model gan_cnn --use-gan-augmentation
```

For NVIDIA GPU support on Linux, install:

```bash
python -m pip install -e ".[tensorflow,pytorch,jax-cuda13]"
```

The TensorFlow CLI configures the pip-installed CUDA library paths before importing TensorFlow.

The processed files in `data/processed/*.npy` are intended to be committed through Git LFS. They are excluded from PyPI builds.

Documentation is configured for GitHub Pages at:

https://mhmdaskari.github.io/EEG_DL_Classification/

This project began as part of UCLA C247, Deep Learning and Neural Networks, Winter 2023. See [Report.pdf](Report.pdf) for the report associated with that work.
