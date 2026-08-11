# Incremental selective-precision test

The **Q4 gate+up blocks 0-2** design was predeclared as a marginal 11.6 MiB extension
to **Q4 gate+up blocks 0-1**. It passes only if its paired per-chunk KL reduction over the
current model has a descriptive interval above zero.

| Model | Role | KL | Recovery from Q2 | JS recovery | Top-1 | Top-10 |
|---|---|---:|---:|---:|---:|---:|
| Q2 baseline | base | 0.2214713 | 0.00% | 0.00% | 77.3% | 75.7% |
| Q4 gate+up blocks 0-1 | current | 0.2103633 | 5.02% | 4.52% | 78.7% | 76.4% |
| Q4 gate+up blocks 0-2 | extension | 0.2096200 | 5.35% | 4.84% | 78.9% | 76.4% |

The extension improves 18/32
chunks versus the current model. Mean incremental KL reduction is
0.0007433, with interval
[-0.0016076,
0.0030942]. It **fails** the predeclared
incremental gate.

The extension raises total recovery by only 0.34 percentage point for 11.6 MiB. Because
the paired incremental interval crosses zero, no code or confirmation captures are run and
blocks 0-1 remain the recommendation.

Aggregate values are in `incremental_precision.csv`; paired chunk values are in
`incremental_precision_chunks.csv`.
