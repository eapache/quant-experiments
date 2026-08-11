# Cumulative-mass calibrated nucleus sampler

The target is BF16 at T=0.8/top-p=0.95. Calibration measures how much
reference probability lies in prefixes sorted by the quant candidate, then inverts
that monotonic curve to choose a candidate top-p. The conditioned model fits separate
curves in quant-entropy bins. Bin count is selected on inner held-out chunks and may
fall back to the global curve. Scalar baselines are the independently cross-validated
settings from the established sampler-grid experiment.

| Method | JS | Recovered | TV | Support Jaccard | Ref mass retained | Mean candidate top-p |
|---|---:|---:|---:|---:|---:|---:|
| same-settings | 0.0663133 | 0.00% | 0.2175111 | 62.2% | 95.93% | 0.9500 |
| tuned-temperature | 0.0648402 | 2.22% | 0.2132031 | 65.4% | 94.71% | 0.9500 |
| tuned-top-p | 0.0658075 | 0.76% | 0.2153180 | 65.8% | 94.41% | 0.9312 |
| tuned-temperature-top-p | 0.0645550 | 2.65% | 0.2130169 | 64.3% | 95.16% | 0.9650 |
| mass-global | 0.0658876 | 0.64% | 0.2160167 | 64.9% | 94.99% | 0.9387 |
| mass-conditioned-forced | 0.0663343 | -0.03% | 0.2166231 | 63.8% | 95.14% | 0.9302 |
| mass-conditioned-selected | 0.0659249 | 0.59% | 0.2159305 | 64.9% | 94.98% | 0.9376 |

## Inner-selected conditioning

| Fold | Entropy groups | Enabled | Candidate top-p values |
|---:|---:|:---:|---|
| 0 | 2 | no | 0.939113 |
| 1 | 2 | yes | 0.926415+0.948746 |
| 2 | 4 | no | 0.938767 |
| 3 | 8 | no | 0.936402 |

Fold-level values are in `mass_calibration.csv`.
