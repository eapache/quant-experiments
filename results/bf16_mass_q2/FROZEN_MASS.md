# Frozen cumulative-mass calibration on code

A single cumulative-mass curve was fitted on all 2,016 prose positions and frozen
before reading the code corpus. It implied candidate top-p=0.938814 at T=0.8
for the BF16 T=0.8/top-p=0.95 target. Scalar baselines are the fold-mean
settings selected by the earlier prose sampler search.

| Method | JS | Recovered | TV | Support Jaccard | Ref mass retained |
|---|---:|---:|---:|---:|---:|
| same-settings | 0.0512854 | 0.00% | 0.1582678 | 72.0% | 96.67% |
| frozen-mass-global | 0.0512570 | 0.06% | 0.1573564 | 73.5% | 95.99% |
| frozen-temperature | 0.0508784 | 0.79% | 0.1561580 | 73.7% | 95.84% |
| frozen-top-p | 0.0513730 | -0.17% | 0.1570756 | 73.8% | 95.59% |
| frozen-joint | 0.0506398 | 1.26% | 0.1560444 | 73.0% | 96.21% |

## Per-chunk paired improvements

Positive values mean lower JS than same settings.

| Method | Mean JS reduction | 95% interval | Improved chunks |
|---|---:|---:|---:|
| frozen-mass-global | 0.0000284 | [-0.0002487, 0.0003055] | 16/32 |
| frozen-temperature | 0.0004071 | [0.0000637, 0.0007504] | 22/32 |
| frozen-top-p | -0.0000875 | [-0.0004087, 0.0002336] | 15/32 |
| frozen-joint | 0.0006456 | [0.0002661, 0.0010251] | 22/32 |

Block results are in `frozen_mass.csv`; per-chunk results are in
`frozen_mass_chunks.csv`.
