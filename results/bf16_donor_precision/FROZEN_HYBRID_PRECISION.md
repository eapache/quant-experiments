# Frozen selective-precision hybrid on code

**early-Q8-2** was selected on prose for size efficiency before any hybrid code
logits were loaded. Other hybrid configurations are diagnostic curve points only.

| Model | Role | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | baseline | 0.1792153 | 0.00% | 0.00% | 84.0% | 75.6% | — | — |
| early-Q4-2 | diagnostic | 0.1718936 | 4.09% | 4.15% | 84.4% | 76.2% | 24/32 | [-0.000608, 0.015251] |
| early-Q8-2 | preselected | 0.1695180 | 5.41% | 5.55% | 84.0% | 76.3% | 27/32 | [0.001434, 0.017960] |
| Q4_K_XL | benchmark | 0.0127692 | 92.87% | 92.45% | 95.5% | 92.3% | 32/32 | [0.147815, 0.185078] |

The preselected model changes code KL from 0.1792153 to
0.1695180, recovering 5.41%. It improves
27/32 chunks; its descriptive chunk-level interval
excludes zero: [0.001434,
0.017960].

The Q4 donor recovers 4.09%, or 75.5% of the Q8 donor's code KL reduction, while requiring
only 36.6% of its added bytes. It improves 24/32 chunks, but its descriptive interval
[-0.000608, 0.015251] narrowly crosses zero. This makes early-Q4-2 a strong memory Pareto
point, while early-Q8-2 has the clearer evidence on this one code file.

Aggregate results are in `frozen_hybrid_precision.csv`; per-chunk metrics are in
`frozen_hybrid_precision_chunks.csv`. All intervals reuse chunks from one code file
and are descriptive rather than evidence across independent code workloads.
