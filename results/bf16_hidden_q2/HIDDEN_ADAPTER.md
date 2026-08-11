# Final-hidden-state residual adapter

The target is BF16; inputs are the 2560-dimensional normalized
final hidden states from Q2_K_XL. The adapter is a kernel-form ridge map to
oracle amplitudes of head-restricted residual directions. Complete context chunks are
held out, and both direction count and ridge penalty include a null option and are selected
only on an inner chunk split.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.2214648 | 0.00% | 0.0510451 | 0.2140513 | 77.3% | 75.7% |
| target-temperature | 0.2196204 | 0.83% | 0.0497565 | 0.2092053 | 77.3% | 75.7% |
| target-temperature+token | 0.2178012 | 1.65% | 0.0497741 | 0.2107479 | 77.8% | 75.9% |
| hidden-adapter | 0.2178012 | 1.65% | 0.0497741 | 0.2107479 | 77.8% | 75.9% |
| hidden-adapter-forced | 0.2313056 | -4.44% | 0.0527723 | 0.2181069 | 77.7% | 75.1% |
| low-rank-oracle-16 | 0.0952049 | 57.01% | 0.0216974 | 0.0964113 | 94.2% | 81.9% |

The null adapter was selected in 4/4 folds. Forced use
changed KL from 0.2178012 to 0.2313056; it is a diagnostic only, not a
deployable correction. This first pass still uses only 2,016 paired positions, so it
does not replace the planned larger-corpus test.

## Inner-selected capacity

| Fold | Selected directions | Selected alpha | Forced directions | Forced alpha |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 4 | 100000 |
| 1 | 0 | 1 | 4 | 100000 |
| 2 | 0 | 1 | 4 | 100000 |
| 3 | 0 | 1 | 16 | 100000 |

Fold-level values are in `hidden_adapter.csv`.
