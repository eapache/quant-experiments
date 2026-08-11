# Frozen mean residual control on code

The all-prose mean BF16-minus-Q2 state vector at layer 3 is frozen
and added after block 2 at strength 1 before loading the Python-code
capture. No code reference states or logits tune the 10 KiB sidecar.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw Q2 | 0.1792153 | 0.00% | 0.0415861 | 0.1649629 | 84.0% | 75.6% |
| Frozen mean layer control | 0.1788410 | 0.21% | 0.0414788 | 0.1646203 | 84.1% | 75.6% |

The correction improves 19/32 chunks.
Mean per-chunk KL reduction is 0.0003743; the descriptive chunk-level t interval is
[-0.0017000, 0.0024486].

Per-chunk metrics are in `frozen_mean_control_chunks.csv`.
