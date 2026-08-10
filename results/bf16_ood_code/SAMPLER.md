# Frozen sampler settings on an out-of-domain code corpus

Settings were selected on `chats/first.txt` and frozen before evaluating the Python
source of the synthetic lab. No code-corpus positions were used for fitting. BF16 uses
T=0.8/top-p=0.95. Values are means across 4 held-out code blocks.

| Quant | Method | T | top-p | JS | Block SD | Recovered | TV |
|---|---|---:|---:|---:|---:|---:|---:|
| Q2_K_XL | same-settings | 0.8000 | 0.9500 | 0.0512854 | 0.0061229 | 0.00% | 0.1582678 |
| Q2_K_XL | frozen-temperature | 0.7625 | 0.9500 | 0.0508784 | 0.0061357 | 0.79% | 0.1561580 |
| Q2_K_XL | frozen-top-p | 0.8000 | 0.9312 | 0.0513730 | 0.0060247 | -0.17% | 0.1570756 |
| Q2_K_XL | frozen-joint | 0.7288 | 0.9650 | 0.0506398 | 0.0062580 | 1.26% | 0.1560444 |
| Q4_K_XL | same-settings | 0.8000 | 0.9500 | 0.0054821 | 0.0004036 | 0.00% | 0.0396272 |
| Q4_K_XL | frozen-temperature | 0.7950 | 0.9500 | 0.0054554 | 0.0004076 | 0.49% | 0.0395596 |
| Q4_K_XL | frozen-top-p | 0.8000 | 0.9500 | 0.0054821 | 0.0004036 | 0.00% | 0.0396272 |
| Q4_K_XL | frozen-joint | 0.7813 | 0.9562 | 0.0054505 | 0.0004110 | 0.58% | 0.0397755 |
| Q8_K_XL | same-settings | 0.8000 | 0.9500 | 0.0005653 | 0.0000804 | 0.00% | 0.0087048 |
| Q8_K_XL | frozen-temperature | 0.8000 | 0.9500 | 0.0005653 | 0.0000804 | 0.00% | 0.0087048 |
| Q8_K_XL | frozen-top-p | 0.8000 | 0.9500 | 0.0005653 | 0.0000804 | 0.00% | 0.0087048 |
| Q8_K_XL | frozen-joint | 0.8000 | 0.9500 | 0.0005653 | 0.0000804 | 0.00% | 0.0087048 |

## Per-chunk paired improvements

Positive values mean lower JS than the same-settings baseline. The interval is
a normal 95% interval over 32 complete 63-position chunks.

| Quant | Method | Mean JS reduction | 95% interval | Improved chunks |
|---|---|---:|---:|---:|
| Q2_K_XL | frozen-temperature | 0.0004071 | [0.0000637, 0.0007504] | 22/32 |
| Q2_K_XL | frozen-top-p | -0.0000875 | [-0.0004087, 0.0002336] | 15/32 |
| Q2_K_XL | frozen-joint | 0.0006456 | [0.0002661, 0.0010251] | 22/32 |
| Q4_K_XL | frozen-temperature | 0.0000267 | [-0.0000215, 0.0000749] | 18/32 |
| Q4_K_XL | frozen-top-p | 0.0000000 | [0.0000000, 0.0000000] | 0/32 |
| Q4_K_XL | frozen-joint | 0.0000316 | [-0.0000484, 0.0001116] | 16/32 |
| Q8_K_XL | frozen-temperature | 0.0000000 | [0.0000000, 0.0000000] | 0/32 |
| Q8_K_XL | frozen-top-p | 0.0000000 | [0.0000000, 0.0000000] | 0/32 |
| Q8_K_XL | frozen-joint | 0.0000000 | [0.0000000, 0.0000000] | 0/32 |

Raw block-level results are in `frozen_sampler.csv`; per-chunk values are in
`frozen_sampler_chunks.csv`.
