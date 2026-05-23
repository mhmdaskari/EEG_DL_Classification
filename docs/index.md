# EEGClassify

`eegclassify` is a reproducible EEG classification package built from the original C247 project notebooks. It keeps the original CNN, LSTM, CNN-LSTM, and GAN-CNN experiments, while moving the reusable code into a Python package with TensorFlow, PyTorch, and JAX/Flax implementations.

The project focuses on:

- explicit BCI Competition IV data set 2a provenance
- repeatable raw-data conversion into the six `.npy` files used by the original notebooks
- object-oriented model, training, and preprocessing code
- notebooks that can run quick smoke checks or full 100-epoch reproductions

The original report remains available as [Report.pdf](https://github.com/mhmdaskari/EEG_DL_Classification/blob/main/Report.pdf), and the original notebook code is preserved under `notebooks/legacy/`.
