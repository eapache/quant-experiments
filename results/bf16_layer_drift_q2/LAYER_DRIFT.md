# Residual-stream quantization drift by layer

Aligned teacher-forced states from BF16 and Q2_K_XL are compared
at the input of every transformer block. Metrics cover the same evaluated positions as
the saved output distributions. Relative error is state-error RMS divided by reference
state RMS. The scale-adjusted column removes one globally fitted scalar only as a
diagnostic. Static-bias recovery uses four held-out blocks of complete context chunks.

Qwen3.5 uses full attention in layers 3, 7, ..., 31; the other layers are recurrent
gated-delta blocks. A row at layer L measures all drift accumulated before block L.

| Layer | Type | Ref RMS | Error RMS | Relative | Scaled relative | Cosine | Static-bias recovery | Error/KL corr. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | recurrent | 0.01381 | 0.00025 | 1.79% | 1.79% | 0.999842 | 0.59% | -0.020 |
| 1 | recurrent | 0.04721 | 0.00535 | 11.33% | 11.33% | 0.993921 | 8.48% | 0.059 |
| 2 | recurrent | 0.05978 | 0.00974 | 16.29% | 16.24% | 0.987387 | 3.84% | 0.043 |
| 3 | full attention | 0.06064 | 0.01343 | 22.14% | 22.11% | 0.975703 | 9.15% | -0.008 |
| 4 | recurrent | 0.08653 | 0.01851 | 21.39% | 21.39% | 0.977079 | 8.38% | -0.027 |
| 5 | recurrent | 0.10307 | 0.02605 | 25.28% | 25.19% | 0.968458 | 2.36% | 0.042 |
| 6 | recurrent | 0.12118 | 0.03380 | 27.89% | 27.85% | 0.961299 | 2.40% | 0.075 |
| 7 | full attention | 0.13869 | 0.04061 | 29.28% | 29.03% | 0.956668 | 2.81% | -0.006 |
| 8 | recurrent | 0.14168 | 0.04698 | 33.16% | 32.96% | 0.943388 | 3.95% | 0.000 |
| 9 | recurrent | 0.14679 | 0.05103 | 34.77% | 34.53% | 0.937970 | 3.11% | 0.008 |
| 10 | recurrent | 0.15685 | 0.05653 | 36.04% | 35.83% | 0.933418 | 2.82% | 0.001 |
| 11 | full attention | 0.17232 | 0.06267 | 36.37% | 36.17% | 0.932628 | 3.33% | -0.013 |
| 12 | recurrent | 0.17459 | 0.06738 | 38.59% | 38.31% | 0.923461 | 3.86% | -0.008 |
| 13 | recurrent | 0.17753 | 0.07019 | 39.54% | 39.19% | 0.919902 | 3.71% | 0.005 |
| 14 | recurrent | 0.18722 | 0.07457 | 39.83% | 39.50% | 0.917930 | 3.22% | 0.007 |
| 15 | full attention | 0.20122 | 0.08084 | 40.18% | 39.87% | 0.915391 | 3.15% | 0.013 |
| 16 | recurrent | 0.21074 | 0.08734 | 41.45% | 41.05% | 0.910450 | 3.23% | 0.019 |
| 17 | recurrent | 0.22008 | 0.09320 | 42.35% | 41.84% | 0.907733 | 3.30% | 0.054 |
| 18 | recurrent | 0.23541 | 0.09659 | 41.03% | 40.59% | 0.912814 | 2.99% | 0.049 |
| 19 | full attention | 0.27882 | 0.10996 | 39.44% | 39.13% | 0.916549 | 3.56% | 0.038 |
| 20 | recurrent | 0.32831 | 0.12542 | 38.20% | 37.80% | 0.923609 | 3.54% | 0.044 |
| 21 | recurrent | 0.37294 | 0.13915 | 37.31% | 37.03% | 0.926327 | 3.37% | 0.045 |
| 22 | recurrent | 0.42181 | 0.15643 | 37.09% | 36.72% | 0.927162 | 3.11% | 0.052 |
| 23 | full attention | 0.50231 | 0.18424 | 36.68% | 36.52% | 0.930214 | 2.97% | 0.059 |
| 24 | recurrent | 0.55049 | 0.20223 | 36.74% | 36.59% | 0.927299 | 3.93% | 0.084 |
| 25 | recurrent | 0.60663 | 0.22051 | 36.35% | 36.19% | 0.928250 | 4.13% | 0.109 |
| 26 | recurrent | 0.65818 | 0.24003 | 36.47% | 36.25% | 0.928186 | 4.16% | 0.132 |
| 27 | full attention | 0.71613 | 0.27556 | 38.48% | 38.33% | 0.926933 | 3.55% | 0.159 |
| 28 | recurrent | 0.76802 | 0.29096 | 37.88% | 37.64% | 0.922395 | 4.82% | 0.199 |
| 29 | recurrent | 0.88624 | 0.32661 | 36.85% | 36.66% | 0.926821 | 4.96% | 0.231 |
| 30 | recurrent | 0.96600 | 0.36478 | 37.76% | 37.57% | 0.924325 | 5.04% | 0.270 |
| 31 | full attention | 1.05717 | 0.42192 | 39.91% | 39.72% | 0.917987 | 5.03% | 0.270 |

The largest adjacent increase in relative error is from layer 0 to 1 (+9.54%). The input
embedding starts at 1.79% error, but drift reaches 22.14% before the first full-attention
block. Relative error peaks at layer 17 (42.35%); by layer 31, its per-row correlation with
final KL is 0.270.

Machine-readable layer summaries are in `layer_drift.csv`; held-out static-bias folds
are in `layer_bias_folds.csv`.
