# Quant sampler compensation lab

This project tests whether a tiny paired-logit calibration set can compensate
for the output-distribution shift introduced by weight quantization.

It compares:

- no correction;
- a fitted temperature/logit scale;
- temperature plus logarithmic rank-bucket biases;
- a monotone isotonic logit map;
- temperature plus a shrunken per-token bias;
- temperature, token bias, and predicted low-rank residual directions.

The final method is the output-space analogue of finding a low-dimensional
direction: SVD finds dominant reference-minus-quant residual directions across
the vocabulary, and a ridge model predicts their amplitude from quant-only
features at inference.

## Run

The environment needs Python 3.11+ with NumPy, SciPy, pandas, matplotlib, and
scikit-learn.

```bash
python3 quant_sampler_lab.py --output-dir results
```

For a shorter smoke test:

```bash
python3 quant_sampler_lab.py --quick --output-dir results-quick
```

The generated `REPORT.md` contains the interpretation and real-model protocol.
CSV files contain trial-level measurements.
