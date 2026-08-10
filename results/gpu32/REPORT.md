# Expanded sparse calibration results

The compact analyzer retains only the union of non-clipped Q8/candidate tokens per
position. Four outer folds hold out complete blocks of context chunks. Temperature
is fitted at the downstream temperature. The entropy-token method learns separate
token biases for low-, middle-, and high-entropy calibration tertiles.

## Q2_K_XL at T=0.8

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.2591990 | 0.00% | 77.3% | 0.012406 |
| 63 | target-temperature | 0.2591655 | 0.01% | 77.3% | 0.012406 |
| 63 | target-temperature+token | 0.2735358 | -5.53% | 77.0% | 0.012406 |
| 63 | target-temperature+entropy-token | 0.2647147 | -2.13% | 77.5% | 0.012406 |
| 126 | identity | 0.2591990 | 0.00% | 77.3% | 0.008683 |
| 126 | target-temperature | 0.2593277 | -0.05% | 77.3% | 0.008683 |
| 126 | target-temperature+token | 0.2685787 | -3.62% | 76.7% | 0.008683 |
| 126 | target-temperature+entropy-token | 0.2653246 | -2.36% | 77.2% | 0.008683 |
| 252 | identity | 0.2591990 | 0.00% | 77.3% | 0.009677 |
| 252 | target-temperature | 0.2592750 | -0.03% | 77.3% | 0.009677 |
| 252 | target-temperature+token | 0.2637707 | -1.76% | 77.9% | 0.009677 |
| 252 | target-temperature+entropy-token | 0.2631216 | -1.51% | 77.5% | 0.009677 |
| 504 | identity | 0.2591990 | 0.00% | 77.3% | 0.010150 |
| 504 | target-temperature | 0.2592823 | -0.03% | 77.3% | 0.010150 |
| 504 | target-temperature+token | 0.2617383 | -0.98% | 78.1% | 0.010150 |
| 504 | target-temperature+entropy-token | 0.2622012 | -1.16% | 77.5% | 0.010150 |
| 1008 | identity | 0.2591990 | 0.00% | 77.3% | 0.007127 |
| 1008 | target-temperature | 0.2593382 | -0.05% | 77.3% | 0.007127 |
| 1008 | target-temperature+token | 0.2590250 | 0.07% | 77.8% | 0.007127 |
| 1008 | target-temperature+entropy-token | 0.2596799 | -0.19% | 77.8% | 0.007127 |
| 1512 | identity | 0.2591990 | 0.00% | 77.3% | 0.003375 |
| 1512 | target-temperature | 0.2592540 | -0.02% | 77.3% | 0.003375 |
| 1512 | target-temperature+token | 0.2578914 | 0.50% | 78.0% | 0.003375 |
| 1512 | target-temperature+entropy-token | 0.2583251 | 0.34% | 77.7% | 0.003375 |

## Q2_K_XL at T=1

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.2222293 | 0.00% | 77.3% | 0.009770 |
| 63 | target-temperature | 0.2205923 | 0.74% | 77.3% | 0.009770 |
| 63 | target-temperature+token | 0.2283563 | -2.76% | 77.6% | 0.009770 |
| 63 | target-temperature+entropy-token | 0.2233105 | -0.49% | 77.4% | 0.009770 |
| 126 | identity | 0.2222293 | 0.00% | 77.3% | 0.013963 |
| 126 | target-temperature | 0.2209838 | 0.56% | 77.3% | 0.013963 |
| 126 | target-temperature+token | 0.2248420 | -1.18% | 77.2% | 0.013963 |
| 126 | target-temperature+entropy-token | 0.2225919 | -0.16% | 77.0% | 0.013963 |
| 252 | identity | 0.2222293 | 0.00% | 77.3% | 0.009689 |
| 252 | target-temperature | 0.2206011 | 0.73% | 77.3% | 0.009689 |
| 252 | target-temperature+token | 0.2219765 | 0.11% | 77.6% | 0.009689 |
| 252 | target-temperature+entropy-token | 0.2214610 | 0.35% | 77.7% | 0.009689 |
| 504 | identity | 0.2222293 | 0.00% | 77.3% | 0.008838 |
| 504 | target-temperature | 0.2203817 | 0.83% | 77.3% | 0.008838 |
| 504 | target-temperature+token | 0.2209892 | 0.56% | 77.7% | 0.008838 |
| 504 | target-temperature+entropy-token | 0.2212521 | 0.44% | 77.6% | 0.008838 |
| 1008 | identity | 0.2222293 | 0.00% | 77.3% | 0.006268 |
| 1008 | target-temperature | 0.2204388 | 0.81% | 77.3% | 0.006268 |
| 1008 | target-temperature+token | 0.2193500 | 1.30% | 77.6% | 0.006268 |
| 1008 | target-temperature+entropy-token | 0.2195312 | 1.21% | 78.0% | 0.006268 |
| 1512 | identity | 0.2222293 | 0.00% | 77.3% | 0.003014 |
| 1512 | target-temperature | 0.2203489 | 0.85% | 77.3% | 0.003014 |
| 1512 | target-temperature+token | 0.2186639 | 1.60% | 77.9% | 0.003014 |
| 1512 | target-temperature+entropy-token | 0.2188657 | 1.51% | 77.6% | 0.003014 |

## Q4_K_XL at T=0.8

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0200073 | 0.00% | 93.2% | 0.006021 |
| 63 | target-temperature | 0.0200044 | 0.01% | 93.2% | 0.006021 |
| 63 | target-temperature+token | 0.0202093 | -1.01% | 93.5% | 0.006021 |
| 63 | target-temperature+entropy-token | 0.0200054 | 0.01% | 93.6% | 0.006021 |
| 126 | identity | 0.0200073 | 0.00% | 93.2% | 0.006640 |
| 126 | target-temperature | 0.0200304 | -0.12% | 93.2% | 0.006640 |
| 126 | target-temperature+token | 0.0201279 | -0.60% | 93.5% | 0.006640 |
| 126 | target-temperature+entropy-token | 0.0200193 | -0.06% | 93.4% | 0.006640 |
| 252 | identity | 0.0200073 | 0.00% | 93.2% | 0.006574 |
| 252 | target-temperature | 0.0200370 | -0.15% | 93.2% | 0.006574 |
| 252 | target-temperature+token | 0.0198342 | 0.87% | 93.5% | 0.006574 |
| 252 | target-temperature+entropy-token | 0.0197403 | 1.33% | 93.6% | 0.006574 |
| 504 | identity | 0.0200073 | 0.00% | 93.2% | 0.003627 |
| 504 | target-temperature | 0.0199967 | 0.05% | 93.2% | 0.003627 |
| 504 | target-temperature+token | 0.0196556 | 1.76% | 93.4% | 0.003627 |
| 504 | target-temperature+entropy-token | 0.0196584 | 1.74% | 93.5% | 0.003627 |
| 1008 | identity | 0.0200073 | 0.00% | 93.2% | 0.002511 |
| 1008 | target-temperature | 0.0200011 | 0.03% | 93.2% | 0.002511 |
| 1008 | target-temperature+token | 0.0195845 | 2.11% | 93.3% | 0.002511 |
| 1008 | target-temperature+entropy-token | 0.0196048 | 2.01% | 93.5% | 0.002511 |
| 1512 | identity | 0.0200073 | 0.00% | 93.2% | 0.001236 |
| 1512 | target-temperature | 0.0199934 | 0.07% | 93.2% | 0.001236 |
| 1512 | target-temperature+token | 0.0195165 | 2.45% | 93.6% | 0.001236 |
| 1512 | target-temperature+entropy-token | 0.0195362 | 2.35% | 93.7% | 0.001236 |

## Q4_K_XL at T=1

| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |
|---:|---|---:|---:|---:|---:|
| 63 | identity | 0.0171895 | 0.00% | 93.2% | 0.002579 |
| 63 | target-temperature | 0.0172093 | -0.12% | 93.2% | 0.002579 |
| 63 | target-temperature+token | 0.0172923 | -0.60% | 93.5% | 0.002579 |
| 63 | target-temperature+entropy-token | 0.0171550 | 0.20% | 93.5% | 0.002579 |
| 126 | identity | 0.0171895 | 0.00% | 93.2% | 0.002387 |
| 126 | target-temperature | 0.0172053 | -0.09% | 93.2% | 0.002387 |
| 126 | target-temperature+token | 0.0172018 | -0.07% | 93.4% | 0.002387 |
| 126 | target-temperature+entropy-token | 0.0171242 | 0.38% | 93.3% | 0.002387 |
| 252 | identity | 0.0171895 | 0.00% | 93.2% | 0.003786 |
| 252 | target-temperature | 0.0172207 | -0.18% | 93.2% | 0.003786 |
| 252 | target-temperature+token | 0.0169947 | 1.13% | 93.5% | 0.003786 |
| 252 | target-temperature+entropy-token | 0.0169521 | 1.38% | 93.6% | 0.003786 |
| 504 | identity | 0.0171895 | 0.00% | 93.2% | 0.002409 |
| 504 | target-temperature | 0.0171990 | -0.06% | 93.2% | 0.002409 |
| 504 | target-temperature+token | 0.0168593 | 1.92% | 93.6% | 0.002409 |
| 504 | target-temperature+entropy-token | 0.0168592 | 1.92% | 93.5% | 0.002409 |
| 1008 | identity | 0.0171895 | 0.00% | 93.2% | 0.001689 |
| 1008 | target-temperature | 0.0172031 | -0.08% | 93.2% | 0.001689 |
| 1008 | target-temperature+token | 0.0167850 | 2.35% | 93.4% | 0.001689 |
| 1008 | target-temperature+entropy-token | 0.0167804 | 2.38% | 93.6% | 0.001689 |
| 1512 | identity | 0.0171895 | 0.00% | 93.2% | 0.000793 |
| 1512 | target-temperature | 0.0171963 | -0.04% | 93.2% | 0.000793 |
| 1512 | target-temperature+token | 0.0167327 | 2.66% | 93.7% | 0.000793 |
| 1512 | target-temperature+entropy-token | 0.0167443 | 2.59% | 93.8% | 0.000793 |

Fold-level values are in `sparse_calibration.csv`.
