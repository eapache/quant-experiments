# Head-restricted low-rank denoiser

The target is BF16. Residual directions are learned only on the union of
the reference and candidate top-128 heads. Each outer fold holds out complete chunks;
the number of directions and ridge penalty are selected on an inner chunk split.
The predicted method receives quant-only distribution features. The oracle uses the
held-out reference to choose amplitudes and is an unattainable diagnostic upper bound.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.2214648 | 0.00% | 0.0510451 | 0.2140513 | 77.3% | 75.7% |
| target-temperature | 0.2196202 | 0.83% | 0.0497578 | 0.2092108 | 77.3% | 75.7% |
| target-temperature+token | 0.2178015 | 1.65% | 0.0497752 | 0.2107532 | 77.8% | 75.9% |
| low-rank-predicted | 0.2182810 | 1.44% | 0.0497486 | 0.2104691 | 78.0% | 75.8% |
| low-rank-oracle-16 | 0.0950399 | 57.09% | 0.0216580 | 0.0958926 | 94.4% | 82.3% |

## Inner-selected capacity

| Fold | Directions | Ridge alpha |
|---:|---:|---:|
| 0 | 0 | 0.1 |
| 1 | 0 | 0.1 |
| 2 | 1 | 1000 |
| 3 | 0 | 0.1 |

Fold-level values are in `low_rank.csv`.
