# Installation

Create and activate an environment, then install the package in editable mode.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[data,notebooks,docs,dev]"
```

Install framework extras as needed:

```bash
python -m pip install -e ".[tensorflow]"
python -m pip install -e ".[pytorch]"
python -m pip install -e ".[jax]"
```

For NVIDIA GPU runs on Linux:

```bash
python -m pip install -e ".[tensorflow,pytorch,jax-cuda13]"
```

The TensorFlow CLI automatically restarts once with the pip-installed NVIDIA
libraries on `LD_LIBRARY_PATH` and `XLA_FLAGS` pointing at CUDA `libdevice`. For
notebooks, start Jupyter from a shell where those variables are already available,
or run the CLI for full GPU reproductions.

For everything in one environment:

```bash
python -m pip install -e ".[all]"
```

For package build checks without uploading to PyPI:

```bash
python -m pip install -e ".[release]"
python -m build
python -m twine check dist/*
```

The processed `.npy` files are kept in Git LFS for repository reproducibility, but
they are excluded from the PyPI source distribution and wheel.
