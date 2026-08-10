# Frozen sampler settings on an out-of-domain code corpus

Settings were selected on `chats/first.txt` and frozen before evaluating the Python
source of the synthetic lab. No code-corpus positions were used for fitting. Q8 uses
T=0.8/top-p=0.95. Values are means across 4 held-out code blocks.

| Quant | Method | T | top-p | JS | Block SD | Recovered | TV |
|---|---|---:|---:|---:|---:|---:|---:|
| Q2_K_XL | same-settings | 0.8000 | 0.9500 | 0.0514493 | 0.0060135 | 0.00% | 0.1585439 |
| Q2_K_XL | frozen-temperature | 0.7600 | 0.9500 | 0.0510595 | 0.0060503 | 0.76% | 0.1563722 |
| Q2_K_XL | frozen-top-p | 0.8000 | 0.9313 | 0.0515347 | 0.0059296 | -0.17% | 0.1573186 |
| Q2_K_XL | frozen-joint | 0.7275 | 0.9650 | 0.0508212 | 0.0061489 | 1.22% | 0.1563816 |
| Q4_K_XL | same-settings | 0.8000 | 0.9500 | 0.0055684 | 0.0004530 | 0.00% | 0.0402880 |
| Q4_K_XL | frozen-temperature | 0.7950 | 0.9500 | 0.0055471 | 0.0004573 | 0.38% | 0.0402323 |
| Q4_K_XL | frozen-joint | 0.7812 | 0.9575 | 0.0054955 | 0.0004876 | 1.31% | 0.0404914 |

## Per-chunk paired improvements

Positive values mean lower JS than the same-settings baseline. The interval is
a normal 95% interval over 32 complete 63-position chunks.

| Quant | Method | Mean JS reduction | 95% interval | Improved chunks |
|---|---|---:|---:|---:|
| Q2_K_XL | frozen-temperature | 0.0003898 | [0.0000264, 0.0007532] | 21/32 |
| Q2_K_XL | frozen-top-p | -0.0000855 | [-0.0003961, 0.0002252] | 15/32 |
| Q2_K_XL | frozen-joint | 0.0006281 | [0.0002455, 0.0010107] | 21/32 |
| Q4_K_XL | frozen-temperature | 0.0000213 | [-0.0000269, 0.0000696] | 18/32 |
| Q4_K_XL | frozen-joint | 0.0000729 | [-0.0000124, 0.0001583] | 18/32 |

Raw block-level results are in `frozen_sampler.csv`; per-chunk values are in
`frozen_sampler_chunks.csv`.
