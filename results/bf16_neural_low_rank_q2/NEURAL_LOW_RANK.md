# Tiny nonlinear low-rank predictor

The target is BF16. A one-hidden-layer tanh network reads quant-only
top-logit gaps/probabilities plus projections of the candidate distribution onto the
learned residual basis, then predicts 16 context-specific correction amplitudes.
Hidden width, weight decay, and training duration are selected on inner held-out
chunks. The selected variant may reject the network in favor of static token bias.
The forced variant reports the best non-null inner configuration on untouched outer
chunks. The oracle remains unattainable and uses each held-out BF16 row.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.2214648 | 0.00% | 0.0510451 | 0.2140513 | 77.3% | 75.7% |
| target-temperature | 0.2196202 | 0.83% | 0.0497578 | 0.2092108 | 77.3% | 75.7% |
| target-temperature+token | 0.2178015 | 1.65% | 0.0497752 | 0.2107532 | 77.8% | 75.9% |
| neural-low-rank-forced | 0.2364235 | -6.75% | 0.0528760 | 0.2162427 | 77.5% | 74.7% |
| neural-low-rank-selected | 0.2178015 | 1.65% | 0.0497752 | 0.2107532 | 77.8% | 75.9% |
| low-rank-oracle-16 | 0.0950399 | 57.09% | 0.0216580 | 0.0958926 | 94.4% | 82.3% |

## Inner-selected configurations

| Fold | Hidden | Parameters | Weight decay | Steps | Enabled |
|---:|---:|---:|---:|---:|:---:|
| 0 | 8 | 1624 | 0.1 | 100 | no |
| 1 | 32 | 6448 | 0.1 | 400 | no |
| 2 | 8 | 1624 | 0.1 | 400 | no |
| 3 | 8 | 1624 | 0.1 | 200 | no |

Fold-level values are in `neural_low_rank.csv`.
