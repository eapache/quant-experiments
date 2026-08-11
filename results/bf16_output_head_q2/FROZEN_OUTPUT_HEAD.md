# Frozen output-head sidecar on code

This test freezes head size 256 and strength 1 before loading the existing
Python-code capture. It applies the exact BF16-minus-Q6_K tied-head correction from
the GGUF weights to the captured Q2 final hidden state. No code reference logits tune
the transform.

The evaluation touches 26,897 distinct top-256 vocabulary rows.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw Q2 | 0.1792153 | 0.00% | 0.0415861 | 0.1649629 | 84.0% | 75.6% |
| Frozen output-head sidecar | 0.1789424 | 0.15% | 0.0415095 | 0.1647065 | 84.1% | 75.6% |

The correction improves 20/32 chunks. Mean
per-chunk KL reduction is 0.0002729; the descriptive chunk-level t interval is
[-0.0001954, 0.0007413].

## Alignment diagnostics

- Recomputed Q6_K-logit centered RMSE: 0.052585
- Output-head correction RMS: 0.035613
- Full BF16-vs-Q2 residual RMS: 0.978618
- Correction/residual correlation: 0.0280

Per-chunk metrics are in `frozen_output_head_chunks.csv`.
