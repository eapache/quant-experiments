# Frozen selective-precision hybrid on code

**Q4-FFN-only** was selected on prose for size efficiency before any hybrid code
logits were loaded. Other hybrid configurations are diagnostic curve points only.

| Model | Role | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | baseline | 0.1792153 | 0.00% | 0.00% | 84.0% | 75.6% | — | — |
| Q4-FFN-only | preselected | 0.1706995 | 4.75% | 4.63% | 84.7% | 76.0% | 28/32 | [0.004977, 0.012055] |
| Q4-full-blocks | diagnostic | 0.1718936 | 4.09% | 4.15% | 84.4% | 76.2% | 24/32 | [-0.000608, 0.015251] |
| Q8-full-blocks | benchmark | 0.1695180 | 5.41% | 5.55% | 84.0% | 76.3% | 27/32 | [0.001434, 0.017960] |

The preselected model changes code KL from 0.1792153 to
0.1706995, recovering 4.75%. It improves
28/32 chunks; its descriptive chunk-level interval
excludes zero: [0.004977,
0.012055].

FFN-only also beats the complete Q4 block upgrade on this code file: 4.75% versus 4.09%
recovery, with 28/32 rather than 24/32 chunks improved. It retains 87.8% of the Q8 block
hybrid's absolute KL reduction while using only 21.2% of its added bytes. A paired
five-repeat CUDA0 benchmark measures 222.54 generation tok/s, 3.25% below Q2; differences
among the hybrid variants are small relative to their run standard deviations.

Aggregate results are in `frozen_hybrid_precision.csv`; per-chunk metrics are in
`frozen_hybrid_precision_chunks.csv`. All intervals reuse chunks from one code file
and are descriptive rather than evidence across independent code workloads.
