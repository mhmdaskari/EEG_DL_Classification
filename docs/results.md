# Results

The original notebooks are the parity reference for the package implementation.
Cross-framework parity means matching experiment semantics and landing in the same
accuracy band, not bitwise-identical weights or histories.

| Experiment | Original notebook accuracy | Parity target |
| --- | ---: | --- |
| CNN | 70.49% | within about ±5 percentage points |
| CNN-LSTM | 60.95% | within about ±5 percentage points |
| LSTM | 39.62% | within about ±5 percentage points |
| GAN+CNN | 68.23% | within about ±5 percentage points |

GAN augmentation can be applied to any classifier, but only GAN+CNN is compared to
the original GAN notebook because the legacy project did not report GAN+LSTM or
GAN+CNN-LSTM baselines.

Local runs write their detailed metrics to `artifacts/runs/`. Summarize those runs
in this page before publishing a release.

## GPU Smoke Runs

On May 23, 2026, fast-dev GPU smoke runs completed on an NVIDIA GeForce RTX 5080
for TensorFlow, PyTorch, and JAX. These runs use one classifier epoch and one GAN
epoch, so they verify execution and device setup rather than final parity accuracy.
The machine-readable summary is stored in `artifacts/gpu_smoke_results.json`.
