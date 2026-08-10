# Held-out ordinary sampler compensation

For each outer fold, 256 evenly spaced positions from the other chunks tune only
the candidate quant's temperature and top-p. The target is the BF16 distribution at
T=0.8/top-p=0.95. All reported divergences are on complete held-out chunk blocks.

| Quant | Method | JS | Recovered | TV | Support Jaccard | Candidate T | Candidate top-p |
|---|---|---:|---:|---:|---:|---:|---:|
| Q2_K_XL | same-settings | 0.0663133 | 0.00% | 0.2175111 | 62.2% | 0.8000 | 0.9500 |
| Q2_K_XL | tuned-temperature | 0.0648402 | 2.22% | 0.2132031 | 65.4% | 0.7625 | 0.9500 |
| Q2_K_XL | tuned-top-p | 0.0658075 | 0.76% | 0.2153180 | 65.8% | 0.8000 | 0.9312 |
| Q2_K_XL | tuned-temperature-top-p | 0.0645550 | 2.65% | 0.2130169 | 64.3% | 0.7288 | 0.9650 |
| Q4_K_XL | same-settings | 0.0072688 | 0.00% | 0.0577350 | 88.1% | 0.8000 | 0.9500 |
| Q4_K_XL | tuned-temperature | 0.0072793 | -0.14% | 0.0577660 | 88.1% | 0.7950 | 0.9500 |
| Q4_K_XL | tuned-top-p | 0.0072688 | 0.00% | 0.0577350 | 88.1% | 0.8000 | 0.9500 |
| Q4_K_XL | tuned-temperature-top-p | 0.0073692 | -1.38% | 0.0587214 | 87.7% | 0.7813 | 0.9562 |
| Q8_K_XL | same-settings | 0.0008728 | 0.00% | 0.0139279 | 96.9% | 0.8000 | 0.9500 |
| Q8_K_XL | tuned-temperature | 0.0008728 | 0.00% | 0.0139279 | 96.9% | 0.8000 | 0.9500 |
| Q8_K_XL | tuned-top-p | 0.0008728 | 0.00% | 0.0139279 | 96.9% | 0.8000 | 0.9500 |
| Q8_K_XL | tuned-temperature-top-p | 0.0008728 | 0.00% | 0.0139279 | 96.9% | 0.8000 | 0.9500 |

## Fold-level tuned parameters

| Quant | Fold | Candidate T | Candidate top-p | Train JS | Test JS |
|---|---:|---:|---:|---:|---:|
| Q2_K_XL | 0 | 0.7400 | 0.9600 | 0.0662671 | 0.0677803 |
| Q2_K_XL | 1 | 0.7450 | 0.9600 | 0.0683705 | 0.0592267 |
| Q2_K_XL | 2 | 0.7150 | 0.9700 | 0.0661295 | 0.0620421 |
| Q2_K_XL | 3 | 0.7150 | 0.9700 | 0.0595006 | 0.0691710 |
| Q4_K_XL | 0 | 0.7800 | 0.9600 | 0.0069280 | 0.0075421 |
| Q4_K_XL | 1 | 0.7950 | 0.9500 | 0.0073558 | 0.0068183 |
| Q4_K_XL | 2 | 0.7800 | 0.9550 | 0.0073250 | 0.0075647 |
| Q4_K_XL | 3 | 0.7700 | 0.9600 | 0.0073202 | 0.0075515 |
| Q8_K_XL | 0 | 0.8000 | 0.9500 | 0.0008586 | 0.0009231 |
| Q8_K_XL | 1 | 0.8000 | 0.9500 | 0.0009557 | 0.0007310 |
| Q8_K_XL | 2 | 0.8000 | 0.9500 | 0.0008859 | 0.0008698 |
| Q8_K_XL | 3 | 0.8000 | 0.9500 | 0.0007330 | 0.0009675 |

Raw fold-level results are in `sampler_grid.csv`.
