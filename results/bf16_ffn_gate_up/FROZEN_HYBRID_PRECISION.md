# Frozen selective-precision hybrid on code

**Q4-gate+up** was selected on prose for size efficiency before any hybrid code
logits were loaded. Other hybrid configurations are diagnostic curve points only.

| Model | Role | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | baseline | 0.1792153 | 0.00% | 0.00% | 84.0% | 75.6% | — | — |
| Q4-up-only | diagnostic | 0.1769131 | 1.28% | 1.67% | 83.9% | 75.5% | 20/32 | [-0.001214, 0.005818] |
| Q4-gate+up | preselected | 0.1737914 | 3.03% | 3.48% | 84.5% | 75.8% | 24/32 | [0.001743, 0.009104] |
| Q4-all-FFN | diagnostic | 0.1706995 | 4.75% | 4.63% | 84.7% | 76.0% | 28/32 | [0.004977, 0.012055] |
| Q8-full-blocks | benchmark | 0.1695180 | 5.41% | 5.55% | 84.0% | 76.3% | 27/32 | [0.001434, 0.017960] |

The preselected model changes code KL from 0.1792153 to
0.1737914, recovering 3.03%. It improves
24/32 chunks; its descriptive chunk-level interval
excludes zero: [0.001743,
0.009104].

Gate+up retains 55.9% of the Q8 block hybrid's absolute code KL reduction while using
11.5% as many added bytes. Up-only falls to 1.28% recovery and its interval crosses zero,
so gate is important for portability even though up was the best prose byte-efficiency
point. All-FFN remains stronger on code at 4.75%, showing some workload-dependent benefit
from down despite its negative isolated prose result.

Five-repeat CUDA0 benchmarking measures gate+up at 232.03 generation tok/s, only 0.76%
below the paired Q2 baseline and within roughly one run standard deviation. Prompt speed is
also unchanged within variation. The compact hybrid has no clearly established throughput
penalty at this measurement resolution.

Aggregate results are in `frozen_hybrid_precision.csv`; per-chunk metrics are in
`frozen_hybrid_precision_chunks.csv`. All intervals reuse chunks from one code file
and are descriptive rather than evidence across independent code workloads.
