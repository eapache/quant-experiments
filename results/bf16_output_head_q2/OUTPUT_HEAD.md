# Sparse high-precision output-head sidecar

The quantized tied embedding/output tensor is `Q6_K` and the reference is
`BF16`. For each row, this experiment adds `(W_ref - W_quant) h` only
to the quant candidate's top-N tokens. N and a blend strength are selected on a chunk
split inside each outer fold. The reference logits in held-out chunks are used only for
evaluation and hyperparameter selection, never to construct the weight correction.

The sweep loaded 26,172 vocabulary rows of 2,560 values. A full
BF16 output sidecar would occupy about 1212.5 MiB before metadata.

## Selected held-out result

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw quant | 0.2214713 | 0.00% | 0.0510455 | 0.2140513 | 77.3% | 75.7% |
| Selected head sidecar | 0.2202139 | 0.57% | 0.0507505 | 0.2131010 | 77.8% | 75.7% |

Fold selections (head, strength): (32, 1.5), (256, 1.5), (64, 2), (256, 1).
The mean per-fold KL reduction is 0.0012574; its four-block t interval is
[0.0003679, 0.0021468].

## Alignment diagnostics

- Recomputed quant-logit centered RMSE: 0.027148
- Output-head correction RMS: 0.036230
- Full BF16-vs-quant residual RMS on the same tokens: 0.958554
- Correction/residual correlation: 0.0371

The first diagnostic verifies that the saved final hidden state feeds the inspected tied
output tensor. Fold-level selections are in `output_head_selected.csv`; the complete
held-out sweep is in `output_head_curves.csv`.
