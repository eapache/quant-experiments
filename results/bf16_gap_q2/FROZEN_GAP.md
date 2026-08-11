# Frozen monotonic gap calibration on code

The scalar temperature, static token bias, and pairwise monotonic mappings were
fitted on all 2,016 prose positions and frozen before loading the code corpus.
The pairwise configuration is head=8, knots=32, rank bins=1,
strength=0.1. It uses the modal discrete capacity and median accepted
strength from the prose inner selections; no code position selected any setting.
The target is BF16 at T=1.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0.1792113 | 0.00% | 0.0415862 | 0.1649629 | 84.0% | 75.6% |
| frozen-target-temperature | 0.1790158 | 0.11% | 0.0407607 | 0.1614883 | 84.0% | 75.6% |
| frozen-temperature+token | 0.1876659 | -4.72% | 0.0430771 | 0.1692101 | 83.1% | 75.6% |
| frozen-pairwise-gap | 0.1876148 | -4.69% | 0.0429896 | 0.1688453 | 83.1% | 75.6% |

## Per-chunk paired improvements over frozen temperature+token

Positive values mean lower KL than the frozen static-bias baseline.

| Method | Mean KL reduction | 95% interval | Improved chunks |
|---|---:|---:|---:|
| identity | 0.0084546 | [0.0056146, 0.0112946] | 30/32 |
| frozen-target-temperature | 0.0086500 | [0.0059054, 0.0113947] | 29/32 |
| frozen-pairwise-gap | 0.0000511 | [-0.0001899, 0.0002920] | 17/32 |

Block results are in `frozen_gap.csv`; per-chunk results are in
`frozen_gap_chunks.csv`.
