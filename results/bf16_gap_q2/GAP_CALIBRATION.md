# Monotonic rank-conditioned gap calibration

The target is BF16. Both transforms start from the fitted scalar
temperature plus static token bias. The direct transform maps gaps from the
candidate's top token to lower-ranked head tokens. The pairwise transform maps all
head-token gaps, then reconstructs a single consistent logit vector. Each mapping is
a weighted monotonic piecewise-linear fit. Head size, knot count, rank-bin count, and
correction strength are selected on an inner chunk split. Selected variants can
reject the transform; forced variants retain the best non-null inner configuration.
All metrics come from untouched outer chunks.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.2214648 | 0.00% | 0.0510451 | 0.2140513 | 77.3% | 75.7% |
| target-temperature | 0.2196202 | 0.83% | 0.0497578 | 0.2092108 | 77.3% | 75.7% |
| target-temperature+token | 0.2178015 | 1.65% | 0.0497752 | 0.2107532 | 77.8% | 75.9% |
| direct-gap-forced | 0.2180506 | 1.54% | 0.0498578 | 0.2111854 | 78.0% | 75.9% |
| direct-gap-selected | 0.2180525 | 1.54% | 0.0498565 | 0.2111520 | 77.9% | 75.9% |
| pairwise-gap-forced | 0.2177665 | 1.67% | 0.0496580 | 0.2103282 | 77.8% | 75.9% |
| pairwise-gap-selected | 0.2177002 | 1.70% | 0.0495843 | 0.2098983 | 77.8% | 75.9% |

## Inner-selected configurations

| Fold | Method | Head | Knots | Rank bins | Strength | Enabled |
|---:|---|---:|---:|---:|---:|:---:|
| 0 | direct-gap-selected | 16 | 32 | 4 | 0.05 | no |
| 0 | pairwise-gap-selected | 8 | 32 | 1 | 0.1 | yes |
| 1 | direct-gap-selected | 16 | 32 | 4 | 0.25 | yes |
| 1 | pairwise-gap-selected | 8 | 32 | 1 | 0.5 | yes |
| 2 | direct-gap-selected | 8 | 8 | 1 | 0.05 | no |
| 2 | pairwise-gap-selected | 8 | 32 | 1 | 0.05 | yes |
| 3 | direct-gap-selected | 32 | 32 | 4 | 0.25 | yes |
| 3 | pairwise-gap-selected | 32 | 4 | 4 | 0.05 | no |

## Paired outer-block improvements over temperature+token

Positive values mean lower KL than the static-bias baseline.

| Method | Mean KL reduction | 95% interval | Improved blocks |
|---|---:|---:|---:|
| direct-gap-selected | -0.0002510 | [-0.0006264, 0.0001245] | 0/4 |
| pairwise-gap-selected | 0.0001013 | [-0.0000102, 0.0002128] | 3/4 |

Fold-level values are in `gap_calibration.csv`.
