# Frozen selective-precision hybrid on code

**Q4-block1-gate+up** was selected on prose for size efficiency before any hybrid code
logits were loaded. Other hybrid configurations are diagnostic curve points only.

| Model | Role | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | baseline | 0.1792153 | 0.00% | 0.00% | 84.0% | 75.6% | — | — |
| Q4-block1-gate+up | preselected | 0.1765193 | 1.50% | 1.71% | 84.1% | 75.7% | 19/32 | [-0.000308, 0.005700] |
| Q4-blocks0+1-gate+up | diagnostic | 0.1737914 | 3.03% | 3.48% | 84.5% | 75.8% | 24/32 | [0.001743, 0.009104] |
| Q4-all-FFN | benchmark | 0.1706995 | 4.75% | 4.63% | 84.7% | 76.0% | 28/32 | [0.004977, 0.012055] |

The preselected model changes code KL from 0.1792153 to
0.1765193, recovering 1.50%. It improves
19/32 chunks; its descriptive chunk-level interval
crosses zero: [-0.000308,
0.005700].

The half-size model retains 49.7% of the two-block code KL reduction for half its bytes,
but improves only 19/32 chunks and does not establish a portable gain. The validated
two-block 21.1 MiB gate+up model remains the smallest supported choice.

A 20-repeat CUDA0 benchmark measures 224.35 generation tok/s for block 1 and 223.27 for
both blocks, versus 225.84 for Q2 (-0.66% and -1.14%). These differences are smaller than
the run standard deviations. Earlier five-repeat absolute rates varied materially with GPU
state, so no reliable speed distinction is claimed at this scale.

Aggregate results are in `frozen_hybrid_precision.csv`; per-chunk metrics are in
`frozen_hybrid_precision_chunks.csv`. All intervals reuse chunks from one code file
and are descriptive rather than evidence across independent code workloads.
