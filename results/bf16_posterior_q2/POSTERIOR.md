# Conditional empirical posterior sampler

The target is BF16. Quant-only entropy, mass, and top-gap features
retrieve calibration rows with similar candidate head shapes. The conditional-mean
method applies their mean rank-wise residual. The posterior-predictive method instead
averages the softmax distributions produced by their correlated residual samples.
Head size and neighbor count are selected separately for each method on an inner
chunk-held-out split. Selected variants may reject the transform and retain the
static-bias baseline; forced variants show the best non-null inner configuration.
All reported metrics come from untouched outer chunks.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.2214648 | 0.00% | 0.0510451 | 0.2140513 | 77.3% | 75.7% |
| target-temperature | 0.2196202 | 0.83% | 0.0497578 | 0.2092108 | 77.3% | 75.7% |
| target-temperature+token | 0.2178015 | 1.65% | 0.0497752 | 0.2107532 | 77.8% | 75.9% |
| conditional-mean-forced | 0.2402280 | -8.47% | 0.0530051 | 0.2155294 | 78.0% | 75.9% |
| conditional-mean-selected | 0.2178015 | 1.65% | 0.0497752 | 0.2107532 | 77.8% | 75.9% |
| posterior-predictive-forced | 0.2226350 | -0.53% | 0.0507886 | 0.2128733 | 78.0% | 75.7% |
| posterior-predictive-selected | 0.2178015 | 1.65% | 0.0497752 | 0.2107532 | 77.8% | 75.9% |

## Inner-selected configurations

| Fold | Method | Head | Neighbors | Enabled |
|---:|---|---:|---:|:---:|
| 0 | conditional-mean-selected | 128 | 128 | no |
| 0 | posterior-predictive-selected | 128 | 128 | no |
| 1 | conditional-mean-selected | 128 | 128 | no |
| 1 | posterior-predictive-selected | 128 | 128 | no |
| 2 | conditional-mean-selected | 128 | 128 | no |
| 2 | posterior-predictive-selected | 128 | 128 | no |
| 3 | conditional-mean-selected | 128 | 128 | no |
| 3 | posterior-predictive-selected | 128 | 128 | no |

Fold-level values are in `posterior.csv`.
