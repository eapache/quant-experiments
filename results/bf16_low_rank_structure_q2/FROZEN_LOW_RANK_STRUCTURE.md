# Frozen prose-to-code low-rank structure check

Every subspace, scalar, and static token bias is fitted on all 2,016 prose positions.
The code capture is loaded only for evaluation. BF16 still supplies per-row oracle
amplitudes to learned and null bases, so this diagnoses structure rather than a
deployable correction.

## Learned versus Gaussian-null rank curve

| Directions | Learned KL | Gaussian KL | Learned advantage | Learned recovery |
|---:|---:|---:|---:|---:|
| 1 | 0.1523397 | 0.1501769 | -0.0021628 | 14.99% |
| 2 | 0.1369415 | 0.1288493 | -0.0080922 | 23.59% |
| 4 | 0.1156910 | 0.1053922 | -0.0102988 | 35.44% |
| 8 | 0.0942480 | 0.0832724 | -0.0109756 | 47.41% |
| 16 | 0.0771405 | 0.0666812 | -0.0104593 | 56.96% |
| 32 | 0.0613929 | 0.0545914 | -0.0068015 | 65.74% |

## Rank-16 controls

| Basis | Oracle KL | Recovered from raw |
|---|---:|---:|
| learned-residual-pca | 0.0771405 | 56.96% |
| gaussian-random | 0.0666812 | 62.79% |
| token-permuted-pca | 0.1053323 | 41.22% |
| sign-randomized-pca | 0.0828407 | 53.77% |

## Per-chunk learned advantage over Gaussian

Mean advantage is -0.0104593 KL with a chunk-level t interval
[-0.0137749, -0.0071436].
Learned PCA wins in 1/32 chunks.
These chunks come from one reused code file, so the interval is descriptive rather
than a document-level generalization interval.

Aggregate metrics are in `frozen_low_rank_structure.csv`; chunk values are in
`frozen_low_rank_structure_chunks.csv`.
