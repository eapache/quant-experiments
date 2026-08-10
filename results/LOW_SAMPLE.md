# Low-sample calibration stability

Each outer fold holds out two complete chunks (126 positions). Calibration uses
1, 2, 4, or 6 other complete chunks (63 to 378 positions). Scalar temperature
is fitted directly at the downstream temperature shown, rather than fitted at T=1
and transferred. Values are means across four outer folds.

## Q2_K_XL at T=0.8

| Calibration positions | Method | KL | Recovered vs identity | Scale mean | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.2771172 | 0.00% | 0.994489 | 0.023499 |
| 63 | target-temperature | 0.2775777 | -0.17% | 0.994489 | 0.023499 |
| 63 | target-temperature+token | 0.2880834 | -3.96% | 0.994489 | 0.023499 |
| 126 | identity | 0.2771172 | 0.00% | 0.994760 | 0.006629 |
| 126 | target-temperature | 0.2771423 | -0.01% | 0.994760 | 0.006629 |
| 126 | target-temperature+token | 0.2818098 | -1.69% | 0.994760 | 0.006629 |
| 252 | identity | 0.2771172 | 0.00% | 0.994558 | 0.004499 |
| 252 | target-temperature | 0.2771412 | -0.01% | 0.994558 | 0.004499 |
| 252 | target-temperature+token | 0.2774750 | -0.13% | 0.994558 | 0.004499 |
| 378 | identity | 0.2771172 | 0.00% | 0.994454 | 0.002228 |
| 378 | target-temperature | 0.2771099 | 0.00% | 0.994454 | 0.002228 |
| 378 | target-temperature+token | 0.2735031 | 1.30% | 0.994454 | 0.002228 |

## Q2_K_XL at T=1

| Calibration positions | Method | KL | Recovered vs identity | Scale mean | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.2444413 | 0.00% | 1.043100 | 0.020262 |
| 63 | target-temperature | 0.2424091 | 0.83% | 1.043100 | 0.020262 |
| 63 | target-temperature+token | 0.2468427 | -0.98% | 1.043100 | 0.020262 |
| 126 | identity | 0.2444413 | 0.00% | 1.035030 | 0.014432 |
| 126 | target-temperature | 0.2424703 | 0.81% | 1.035030 | 0.014432 |
| 126 | target-temperature+token | 0.2440388 | 0.16% | 1.035030 | 0.014432 |
| 252 | identity | 0.2444413 | 0.00% | 1.034184 | 0.008937 |
| 252 | target-temperature | 0.2423756 | 0.85% | 1.034184 | 0.008937 |
| 252 | target-temperature+token | 0.2410028 | 1.41% | 1.034184 | 0.008937 |
| 378 | identity | 0.2444413 | 0.00% | 1.033659 | 0.004377 |
| 378 | target-temperature | 0.2422051 | 0.91% | 1.033659 | 0.004377 |
| 378 | target-temperature+token | 0.2384239 | 2.46% | 1.033659 | 0.004377 |

## Q4_K_XL at T=0.8

| Calibration positions | Method | KL | Recovered vs identity | Scale mean | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0194375 | 0.00% | 0.993140 | 0.002776 |
| 63 | target-temperature | 0.0194612 | -0.12% | 0.993140 | 0.002776 |
| 63 | target-temperature+token | 0.0193391 | 0.51% | 0.993140 | 0.002776 |
| 126 | identity | 0.0194375 | 0.00% | 0.998621 | 0.003202 |
| 126 | target-temperature | 0.0194475 | -0.05% | 0.998621 | 0.003202 |
| 126 | target-temperature+token | 0.0192678 | 0.87% | 0.998621 | 0.003202 |
| 252 | identity | 0.0194375 | 0.00% | 0.998366 | 0.002133 |
| 252 | target-temperature | 0.0194465 | -0.05% | 0.998366 | 0.002133 |
| 252 | target-temperature+token | 0.0190496 | 2.00% | 0.998366 | 0.002133 |
| 378 | identity | 0.0194375 | 0.00% | 0.998230 | 0.001005 |
| 378 | target-temperature | 0.0194400 | -0.01% | 0.998230 | 0.001005 |
| 378 | target-temperature+token | 0.0189168 | 2.68% | 0.998230 | 0.001005 |

## Q4_K_XL at T=1

| Calibration positions | Method | KL | Recovered vs identity | Scale mean | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0171012 | 0.00% | 0.997094 | 0.005404 |
| 63 | target-temperature | 0.0171692 | -0.40% | 0.997094 | 0.005404 |
| 63 | target-temperature+token | 0.0170082 | 0.54% | 0.997094 | 0.005404 |
| 126 | identity | 0.0171012 | 0.00% | 1.001223 | 0.002417 |
| 126 | target-temperature | 0.0171053 | -0.02% | 1.001223 | 0.002417 |
| 126 | target-temperature+token | 0.0169064 | 1.14% | 1.001223 | 0.002417 |
| 252 | identity | 0.0171012 | 0.00% | 1.001321 | 0.001662 |
| 252 | target-temperature | 0.0171099 | -0.05% | 1.001321 | 0.001662 |
| 252 | target-temperature+token | 0.0167266 | 2.19% | 1.001321 | 0.001662 |
| 378 | identity | 0.0171012 | 0.00% | 1.001374 | 0.000781 |
| 378 | target-temperature | 0.0171037 | -0.02% | 1.001374 | 0.000781 |
| 378 | target-temperature+token | 0.0166132 | 2.85% | 1.001374 | 0.000781 |

The repeated identity rows are intentional: they give the exactly matched held-out
baseline for each fold/sample-size comparison. Fold-level results are in
`low_sample.csv`.
