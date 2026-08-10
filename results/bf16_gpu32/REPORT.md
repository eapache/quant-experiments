# Expanded sparse calibration results

The compact analyzer retains only the union of non-clipped BF16/candidate tokens per
position. Four outer folds hold out complete blocks of context chunks. Temperature
is fitted at the downstream temperature. The entropy-token method learns separate
token biases for low-, middle-, and high-entropy calibration tertiles.

## Q2_K_XL at T=0.8

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.2585103 | 0.00% | 77.3% | 0.013083 |
| 63 | target-temperature | 0.2585258 | -0.01% | 77.3% | 0.013083 |
| 63 | target-temperature+token | 0.2725877 | -5.45% | 77.1% | 0.013083 |
| 63 | target-temperature+entropy-token | 0.2637319 | -2.02% | 77.7% | 0.013083 |
| 126 | identity | 0.2585103 | 0.00% | 77.3% | 0.009764 |
| 126 | target-temperature | 0.2586650 | -0.06% | 77.3% | 0.009764 |
| 126 | target-temperature+token | 0.2672945 | -3.40% | 76.9% | 0.009764 |
| 126 | target-temperature+entropy-token | 0.2640487 | -2.14% | 77.3% | 0.009764 |
| 252 | identity | 0.2585103 | 0.00% | 77.3% | 0.009901 |
| 252 | target-temperature | 0.2585969 | -0.03% | 77.3% | 0.009901 |
| 252 | target-temperature+token | 0.2627195 | -1.63% | 77.9% | 0.009901 |
| 252 | target-temperature+entropy-token | 0.2621077 | -1.39% | 77.4% | 0.009901 |
| 504 | identity | 0.2585103 | 0.00% | 77.3% | 0.010026 |
| 504 | target-temperature | 0.2585926 | -0.03% | 77.3% | 0.010026 |
| 504 | target-temperature+token | 0.2607926 | -0.88% | 78.1% | 0.010026 |
| 504 | target-temperature+entropy-token | 0.2611743 | -1.03% | 77.7% | 0.010026 |
| 1008 | identity | 0.2585103 | 0.00% | 77.3% | 0.007045 |
| 1008 | target-temperature | 0.2586472 | -0.05% | 77.3% | 0.007045 |
| 1008 | target-temperature+token | 0.2581743 | 0.13% | 77.8% | 0.007045 |
| 1008 | target-temperature+entropy-token | 0.2588188 | -0.12% | 77.9% | 0.007045 |
| 1512 | identity | 0.2585103 | 0.00% | 77.3% | 0.003341 |
| 1512 | target-temperature | 0.2585651 | -0.02% | 77.3% | 0.003341 |
| 1512 | target-temperature+token | 0.2570195 | 0.58% | 78.0% | 0.003341 |
| 1512 | target-temperature+entropy-token | 0.2574420 | 0.41% | 77.8% | 0.003341 |

## Q2_K_XL at T=1

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.2214713 | 0.00% | 77.3% | 0.010851 |
| 63 | target-temperature | 0.2199135 | 0.70% | 77.3% | 0.010851 |
| 63 | target-temperature+token | 0.2275435 | -2.74% | 77.6% | 0.010851 |
| 63 | target-temperature+entropy-token | 0.2224807 | -0.46% | 77.5% | 0.010851 |
| 126 | identity | 0.2214713 | 0.00% | 77.3% | 0.014103 |
| 126 | target-temperature | 0.2202496 | 0.55% | 77.3% | 0.014103 |
| 126 | target-temperature+token | 0.2238286 | -1.06% | 77.3% | 0.014103 |
| 126 | target-temperature+entropy-token | 0.2215573 | -0.04% | 77.2% | 0.014103 |
| 252 | identity | 0.2214713 | 0.00% | 77.3% | 0.009483 |
| 252 | target-temperature | 0.2198644 | 0.73% | 77.3% | 0.009483 |
| 252 | target-temperature+token | 0.2210279 | 0.20% | 77.9% | 0.009483 |
| 252 | target-temperature+entropy-token | 0.2205095 | 0.43% | 77.8% | 0.009483 |
| 504 | identity | 0.2214713 | 0.00% | 77.3% | 0.008759 |
| 504 | target-temperature | 0.2196478 | 0.82% | 77.3% | 0.008759 |
| 504 | target-temperature+token | 0.2200727 | 0.63% | 78.0% | 0.008759 |
| 504 | target-temperature+entropy-token | 0.2203238 | 0.52% | 77.7% | 0.008759 |
| 1008 | identity | 0.2214713 | 0.00% | 77.3% | 0.006214 |
| 1008 | target-temperature | 0.2197042 | 0.80% | 77.3% | 0.006214 |
| 1008 | target-temperature+token | 0.2185051 | 1.34% | 77.9% | 0.006214 |
| 1008 | target-temperature+entropy-token | 0.2186739 | 1.26% | 78.2% | 0.006214 |
| 1512 | identity | 0.2214713 | 0.00% | 77.3% | 0.002991 |
| 1512 | target-temperature | 0.2196161 | 0.84% | 77.3% | 0.002991 |
| 1512 | target-temperature+token | 0.2178065 | 1.65% | 77.8% | 0.002991 |
| 1512 | target-temperature+entropy-token | 0.2179896 | 1.57% | 77.7% | 0.002991 |

## Q4_K_XL at T=0.8

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0192644 | 0.00% | 93.6% | 0.006128 |
| 63 | target-temperature | 0.0192666 | -0.01% | 93.6% | 0.006128 |
| 63 | target-temperature+token | 0.0194869 | -1.16% | 94.1% | 0.006128 |
| 63 | target-temperature+entropy-token | 0.0192492 | 0.08% | 93.9% | 0.006128 |
| 126 | identity | 0.0192644 | 0.00% | 93.6% | 0.006282 |
| 126 | target-temperature | 0.0192876 | -0.12% | 93.6% | 0.006282 |
| 126 | target-temperature+token | 0.0193325 | -0.35% | 94.0% | 0.006282 |
| 126 | target-temperature+entropy-token | 0.0192111 | 0.28% | 94.0% | 0.006282 |
| 252 | identity | 0.0192644 | 0.00% | 93.6% | 0.005736 |
| 252 | target-temperature | 0.0192870 | -0.12% | 93.6% | 0.005736 |
| 252 | target-temperature+token | 0.0190396 | 1.17% | 94.0% | 0.005736 |
| 252 | target-temperature+entropy-token | 0.0189414 | 1.68% | 94.0% | 0.005736 |
| 504 | identity | 0.0192644 | 0.00% | 93.6% | 0.002860 |
| 504 | target-temperature | 0.0192508 | 0.07% | 93.6% | 0.002860 |
| 504 | target-temperature+token | 0.0188775 | 2.01% | 93.8% | 0.002860 |
| 504 | target-temperature+entropy-token | 0.0188775 | 2.01% | 94.0% | 0.002860 |
| 1008 | identity | 0.0192644 | 0.00% | 93.6% | 0.001925 |
| 1008 | target-temperature | 0.0192523 | 0.06% | 93.6% | 0.001925 |
| 1008 | target-temperature+token | 0.0187821 | 2.50% | 93.8% | 0.001925 |
| 1008 | target-temperature+entropy-token | 0.0188036 | 2.39% | 93.8% | 0.001925 |
| 1512 | identity | 0.0192644 | 0.00% | 93.6% | 0.000972 |
| 1512 | target-temperature | 0.0192482 | 0.08% | 93.6% | 0.000972 |
| 1512 | target-temperature+token | 0.0187283 | 2.78% | 94.1% | 0.000972 |
| 1512 | target-temperature+entropy-token | 0.0187437 | 2.70% | 94.0% | 0.000972 |

## Q4_K_XL at T=1

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0165486 | 0.00% | 93.6% | 0.002893 |
| 63 | target-temperature | 0.0165667 | -0.11% | 93.6% | 0.002893 |
| 63 | target-temperature+token | 0.0166681 | -0.72% | 93.9% | 0.002893 |
| 63 | target-temperature+entropy-token | 0.0165124 | 0.22% | 93.8% | 0.002893 |
| 126 | identity | 0.0165486 | 0.00% | 93.6% | 0.001846 |
| 126 | target-temperature | 0.0165605 | -0.07% | 93.6% | 0.001846 |
| 126 | target-temperature+token | 0.0165198 | 0.17% | 93.8% | 0.001846 |
| 126 | target-temperature+entropy-token | 0.0164339 | 0.69% | 93.8% | 0.001846 |
| 252 | identity | 0.0165486 | 0.00% | 93.6% | 0.003124 |
| 252 | target-temperature | 0.0165735 | -0.15% | 93.6% | 0.003124 |
| 252 | target-temperature+token | 0.0163274 | 1.34% | 93.9% | 0.003124 |
| 252 | target-temperature+entropy-token | 0.0162632 | 1.72% | 93.9% | 0.003124 |
| 504 | identity | 0.0165486 | 0.00% | 93.6% | 0.001777 |
| 504 | target-temperature | 0.0165531 | -0.03% | 93.6% | 0.001777 |
| 504 | target-temperature+token | 0.0162014 | 2.10% | 94.0% | 0.001777 |
| 504 | target-temperature+entropy-token | 0.0161985 | 2.12% | 93.8% | 0.001777 |
| 1008 | identity | 0.0165486 | 0.00% | 93.6% | 0.001250 |
| 1008 | target-temperature | 0.0165554 | -0.04% | 93.6% | 0.001250 |
| 1008 | target-temperature+token | 0.0161162 | 2.61% | 93.9% | 0.001250 |
| 1008 | target-temperature+entropy-token | 0.0161066 | 2.67% | 93.9% | 0.001250 |
| 1512 | identity | 0.0165486 | 0.00% | 93.6% | 0.000586 |
| 1512 | target-temperature | 0.0165517 | -0.02% | 93.6% | 0.000586 |
| 1512 | target-temperature+token | 0.0160727 | 2.88% | 94.1% | 0.000586 |
| 1512 | target-temperature+entropy-token | 0.0160857 | 2.80% | 94.1% | 0.000586 |

## Q8_K_XL at T=0.8

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0011146 | 0.00% | 98.0% | 0.001489 |
| 63 | target-temperature | 0.0011170 | -0.22% | 98.0% | 0.001489 |
| 63 | target-temperature+token | 0.0011391 | -2.20% | 98.1% | 0.001489 |
| 63 | target-temperature+entropy-token | 0.0011291 | -1.30% | 98.1% | 0.001489 |
| 126 | identity | 0.0011146 | 0.00% | 98.0% | 0.002056 |
| 126 | target-temperature | 0.0011197 | -0.45% | 98.0% | 0.002056 |
| 126 | target-temperature+token | 0.0011958 | -7.28% | 98.2% | 0.002056 |
| 126 | target-temperature+entropy-token | 0.0011619 | -4.24% | 98.1% | 0.002056 |
| 252 | identity | 0.0011146 | 0.00% | 98.0% | 0.001632 |
| 252 | target-temperature | 0.0011172 | -0.23% | 98.0% | 0.001632 |
| 252 | target-temperature+token | 0.0011476 | -2.95% | 98.1% | 0.001632 |
| 252 | target-temperature+entropy-token | 0.0011358 | -1.90% | 98.0% | 0.001632 |
| 504 | identity | 0.0011146 | 0.00% | 98.0% | 0.001108 |
| 504 | target-temperature | 0.0011156 | -0.09% | 98.0% | 0.001108 |
| 504 | target-temperature+token | 0.0011287 | -1.26% | 98.2% | 0.001108 |
| 504 | target-temperature+entropy-token | 0.0011287 | -1.27% | 98.1% | 0.001108 |
| 1008 | identity | 0.0011146 | 0.00% | 98.0% | 0.000764 |
| 1008 | target-temperature | 0.0011160 | -0.12% | 98.0% | 0.000764 |
| 1008 | target-temperature+token | 0.0011272 | -1.13% | 98.2% | 0.000764 |
| 1008 | target-temperature+entropy-token | 0.0011264 | -1.05% | 98.2% | 0.000764 |
| 1512 | identity | 0.0011146 | 0.00% | 98.0% | 0.000361 |
| 1512 | target-temperature | 0.0011151 | -0.05% | 98.0% | 0.000361 |
| 1512 | target-temperature+token | 0.0011248 | -0.91% | 98.1% | 0.000361 |
| 1512 | target-temperature+entropy-token | 0.0011259 | -1.01% | 98.0% | 0.000361 |

## Q8_K_XL at T=1

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0010029 | 0.00% | 98.0% | 0.001027 |
| 63 | target-temperature | 0.0010038 | -0.09% | 98.0% | 0.001027 |
| 63 | target-temperature+token | 0.0010194 | -1.64% | 98.0% | 0.001027 |
| 63 | target-temperature+entropy-token | 0.0010122 | -0.93% | 98.1% | 0.001027 |
| 126 | identity | 0.0010029 | 0.00% | 98.0% | 0.001718 |
| 126 | target-temperature | 0.0010077 | -0.48% | 98.0% | 0.001718 |
| 126 | target-temperature+token | 0.0010633 | -6.02% | 98.2% | 0.001718 |
| 126 | target-temperature+entropy-token | 0.0010466 | -4.36% | 98.0% | 0.001718 |
| 252 | identity | 0.0010029 | 0.00% | 98.0% | 0.001268 |
| 252 | target-temperature | 0.0010049 | -0.20% | 98.0% | 0.001268 |
| 252 | target-temperature+token | 0.0010259 | -2.29% | 98.1% | 0.001268 |
| 252 | target-temperature+entropy-token | 0.0010201 | -1.72% | 98.0% | 0.001268 |
| 504 | identity | 0.0010029 | 0.00% | 98.0% | 0.000879 |
| 504 | target-temperature | 0.0010035 | -0.06% | 98.0% | 0.000879 |
| 504 | target-temperature+token | 0.0010118 | -0.89% | 98.2% | 0.000879 |
| 504 | target-temperature+entropy-token | 0.0010129 | -1.00% | 98.2% | 0.000879 |
| 1008 | identity | 0.0010029 | 0.00% | 98.0% | 0.000593 |
| 1008 | target-temperature | 0.0010038 | -0.09% | 98.0% | 0.000593 |
| 1008 | target-temperature+token | 0.0010102 | -0.72% | 98.2% | 0.000593 |
| 1008 | target-temperature+entropy-token | 0.0010112 | -0.83% | 98.3% | 0.000593 |
| 1512 | identity | 0.0010029 | 0.00% | 98.0% | 0.000284 |
| 1512 | target-temperature | 0.0010030 | -0.01% | 98.0% | 0.000284 |
| 1512 | target-temperature+token | 0.0010081 | -0.51% | 98.1% | 0.000284 |
| 1512 | target-temperature+entropy-token | 0.0010094 | -0.65% | 98.0% | 0.000284 |

Fold-level values are in `sparse_calibration.csv`.
