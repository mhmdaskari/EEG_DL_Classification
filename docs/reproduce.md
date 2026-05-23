# Reproduce Results

The package notebooks and CLI share the same experiment settings, so you can move between an interactive workflow and a scripted run without changing the core configuration.

The rendered notebook examples are:

- [Data postprocessing](examples/00_data_postprocessing.ipynb)
- [TensorFlow reproduction](examples/01_tensorflow_reproduction.ipynb)
- [PyTorch reproduction](examples/02_pytorch_reproduction.ipynb)
- [JAX/Flax reproduction](examples/03_jax_reproduction.ipynb)

Each training notebook defaults to:

```python
FAST_DEV_RUN = True
USE_GAN_AUGMENTATION = False
```

This runs a short smoke check. Set `FAST_DEV_RUN = False` for the full 100-epoch
classifier reproduction. Set `USE_GAN_AUGMENTATION = True` to train a conditional
GAN on the training split, append generated samples, and then train the selected
classifier on real plus synthetic training data.

The default experiment settings are:

- labels `769-772` are converted to `0-3`
- validation ratio is `0.17`
- seed is `1`
- batch size is `64`
- classifier epochs are `100`
- GAN epochs are `20`
- augmentation uses trim, max-pooling, averaging with Gaussian noise, and subsampling

GAN augmentation is available for CNN, LSTM, and CNN-LSTM. The GAN+CNN row below is
the canonical GAN-augmented benchmark path for the package.

Reference accuracy targets:

| Experiment | Reference test accuracy |
| --- | ---: |
| CNN | 70.49% |
| CNN-LSTM | 60.95% |
| LSTM | 39.62% |
| GAN+CNN | 68.23% |

Run the CLI equivalent:

```bash
python -m eegclassify.cli train --framework tensorflow --model cnn --fast-dev-run
python -m eegclassify.cli train --framework pytorch --model cnn --fast-dev-run
python -m eegclassify.cli train --framework jax --model cnn --fast-dev-run
```

For GPU runs, install the GPU extras from the installation page and do not set
`CUDA_VISIBLE_DEVICES=-1`. TensorFlow GPU training uses the CLI's CUDA path setup;
PyTorch and JAX use their framework CUDA packages directly.

For the GAN-augmented CNN path:

```bash
python -m eegclassify.cli train \
  --framework tensorflow \
  --model gan_cnn \
  --use-gan-augmentation
```

Each CLI run writes a metrics report under `artifacts/runs/`.

Run tests before long experiments:

```bash
pytest
```

Build docs locally:

```bash
mkdocs build --strict
```
