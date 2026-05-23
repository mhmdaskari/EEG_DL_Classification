# EEGClassify

`eegclassify` is a standalone Python package for reproducible EEG task classification. It combines BCI Competition IV-2a data preparation, experiment-ready preprocessing, and deep learning classifiers across TensorFlow, PyTorch, and JAX/Flax.

The package is designed for users who want to regenerate the processed dataset, run quick framework smoke tests, and scale the same configuration into full training runs.

## What You Can Do

- Download and convert public BCI Competition IV-2a raw data.
- Load the packaged processed `.npy` arrays for local examples.
- Train CNN, LSTM, and CNN-LSTM classifiers in three frameworks.
- Add conditional GAN-generated samples to the training split.
- Compare local runs against the package reference targets.
- Use notebooks for guided workflows or the CLI for repeatable runs.

## Start Here

1. Install the package with the extras you need in [Installation](installation.md).
2. Download or regenerate the EEG arrays in [Data](data.md).
3. Walk through the rendered [Notebook Examples](examples/index.md).
4. Run CLI or notebook experiments with [Reproduce Results](reproduce.md).
5. Check target metrics in [Results](results.md).
6. Build on the package through the [API Reference](api/index.md).

The default examples use `FAST_DEV_RUN=True`, so they are meant to be approachable first and expandable when you are ready for longer runs.
