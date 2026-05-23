# Results

The table below lists package reference targets for the default EEG classification
experiments. Cross-framework consistency means matching experiment semantics and
landing in the same accuracy band, not bitwise-identical weights or histories.

| Experiment | Reference accuracy | Acceptance band |
| --- | ---: | --- |
| CNN | 70.49% | within about ±5 percentage points |
| CNN-LSTM | 60.95% | within about ±5 percentage points |
| LSTM | 39.62% | within about ±5 percentage points |
| GAN+CNN | 68.23% | within about ±5 percentage points |

GAN augmentation can be applied to any classifier. The reference table includes
GAN+CNN as the canonical GAN-augmented benchmark path.

Local runs write their detailed metrics to `artifacts/runs/`. Summarize those runs
in this page before publishing a release.

## GPU Smoke Runs

On May 23, 2026, fast-dev GPU smoke runs completed on an NVIDIA GeForce RTX 5080
for TensorFlow, PyTorch, and JAX. These runs use one classifier epoch and one GAN
epoch, so they verify execution and device setup rather than final parity accuracy.
The machine-readable summary is stored in `artifacts/gpu_smoke_results.json`.
