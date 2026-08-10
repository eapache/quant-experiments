# Held-out ordinary sampler compensation

For each outer fold, 256 evenly spaced positions from the other chunks tune only
the candidate quant's temperature and top-p. The target is the Q8 distribution at
T=0.8/top-p=0.95. All reported divergences are on complete held-out chunk blocks.

| Quant | Method | JS | Recovered | TV | Support Jaccard | Candidate T | Candidate top-p |
|---|---|---:|---:|---:|---:|---:|---:|
| Q2_K_XL | same-settings | 0.0663932 | 0.00% | 0.2177722 | 62.2% | 0.8000 | 0.9500 |
| Q2_K_XL | tuned-temperature | 0.0649256 | 2.21% | 0.2133221 | 65.5% | 0.7600 | 0.9500 |
| Q2_K_XL | tuned-top-p | 0.0658770 | 0.78% | 0.2155578 | 65.8% | 0.8000 | 0.9312 |
| Q2_K_XL | tuned-temperature-top-p | 0.0646430 | 2.64% | 0.2131626 | 64.4% | 0.7275 | 0.9650 |
| Q4_K_XL | same-settings | 0.0075621 | 0.00% | 0.0592919 | 87.8% | 0.8000 | 0.9500 |
| Q4_K_XL | tuned-temperature | 0.0075797 | -0.23% | 0.0593381 | 87.7% | 0.7950 | 0.9500 |
| Q4_K_XL | tuned-top-p | 0.0075621 | 0.00% | 0.0592919 | 87.8% | 0.8000 | 0.9500 |
| Q4_K_XL | tuned-temperature-top-p | 0.0076032 | -0.54% | 0.0601799 | 87.5% | 0.7813 | 0.9575 |

## Fold-level tuned parameters

| Quant | Fold | Candidate T | Candidate top-p | Train JS | Test JS |
|---|---:|---:|---:|---:|---:|
| Q2_K_XL | 0 | 0.7400 | 0.9600 | 0.0658185 | 0.0682292 |
| Q2_K_XL | 1 | 0.7450 | 0.9600 | 0.0684989 | 0.0588988 |
| Q2_K_XL | 2 | 0.7100 | 0.9700 | 0.0660875 | 0.0619742 |
| Q2_K_XL | 3 | 0.7150 | 0.9700 | 0.0595067 | 0.0694699 |
| Q4_K_XL | 0 | 0.7800 | 0.9600 | 0.0071690 | 0.0079756 |
| Q4_K_XL | 1 | 0.7950 | 0.9500 | 0.0077558 | 0.0070282 |
| Q4_K_XL | 2 | 0.7750 | 0.9600 | 0.0076242 | 0.0077224 |
| Q4_K_XL | 3 | 0.7750 | 0.9600 | 0.0076109 | 0.0076867 |

Raw fold-level results are in `sampler_grid.csv`.
