# Controlled low-rank residual diagnostic

Every basis is constructed without the outer test chunks. The oracle then receives
the held-out BF16 row and optimizes the same number of per-row amplitudes for learned
and control bases. Controls are Gaussian random subspaces, token-permuted learned PCA
directions, and PCA after independently randomizing every training residual sign.

## Learned basis versus Gaussian-null rank curve

| Directions | Learned KL | Gaussian-null KL | Learned advantage | Learned recovery | Top-1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1958070 | 0.1955216 | -0.0002855 | 11.59% | 81.8% |
| 2 | 0.1780480 | 0.1790250 | 0.0009770 | 19.60% | 83.3% |
| 4 | 0.1537440 | 0.1555497 | 0.0018058 | 30.58% | 87.5% |
| 8 | 0.1238309 | 0.1272242 | 0.0033933 | 44.09% | 90.3% |
| 16 | 0.0950399 | 0.0989945 | 0.0039546 | 57.09% | 94.4% |
| 32 | 0.0721075 | 0.0746878 | 0.0025804 | 67.44% | 97.2% |

## Rank-16 matched controls

Control SD/range is across seeds after averaging the four outer folds.

| Basis | Oracle KL | Seed SD | Seed range | Recovered from raw |
|---|---:|---:|---:|---:|
| learned-residual-pca | 0.0950399 | — | — | 57.09% |
| gaussian-random | 0.0989945 | 0.0003730 | [0.0986689, 0.0994850] | 55.30% |
| token-permuted-pca | 0.1447358 | 0.0010369 | [0.1437631, 0.1457532] | 34.65% |
| sign-randomized-pca | 0.1067091 | 0.0006926 | [0.1057394, 0.1072136] | 51.82% |

## Distribution concentration

| Distribution | Effective support mean | Effective support median | Mean top-16 mass | Mean top-32 mass |
|---|---:|---:|---:|---:|
| BF16 | 31.5 | 9.7 | 85.9% | 90.1% |
| Q2 | 45.5 | 15.1 | 82.0% | 87.1% |

## Interpretation

The learned rank-16 basis has KL 0.0950399; the best
aggregate control seed has KL 0.0986689. It beats every matched
control in every outer fold: yes.
Learned singular values are 1.66× the sign-null value for component 1 and
1.12× for component 16 (all-component mean 1.31×).
However, the learned basis improves over the mean Gaussian oracle by only
0.0039546 KL, 3.1% of its total
oracle reduction. Its four-block t interval is [-0.0004270,
0.0083363], so the learned advantage is not
independently established at the document/workload level.
The rank curve has no sharp low-dimensional cutoff and the Gaussian curve tracks it
closely. Thus the results suggest some transferable residual covariance, but the prior
57% figure mostly measures generic per-row oracle flexibility on a
concentrated distribution; it is not evidence that the residual is intrinsically
16-dimensional. None of these controls make the oracle deployable or show that its
amplitudes are predictable from quant-only inputs.

Each null family used 4 fixed seeds. Fold-level metrics are in
`low_rank_structure.csv`; singular-value comparisons are in `low_rank_spectrum.csv`,
and concentration diagnostics are in `distribution_support.csv`.

The companion [frozen prose-to-code check](FROZEN_LOW_RANK_STRUCTURE.md) reverses the small
learned advantage: Gaussian directions beat learned PCA at every tested rank and in 31/32
rank-16 code chunks.
