# Selective early-block precision

The hybrids retain every tensor byte from Q2_K_XL except complete early recurrent
blocks copied from a shape-identical higher-precision GGUF. The builder reopens each model
and verifies every raw tensor payload against its intended source. This experiment was
chosen from residual-state localization without inspecting hybrid output logits.

| Model | Size | Added | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | 1.808 GiB | +0.0 MiB | 0.2214713 | 0.00% | 0.00% | 77.3% | 75.7% | — | — |
| early-Q4-2 | 1.873 GiB | +66.9 MiB | 0.2063805 | 6.81% | 6.29% | 78.7% | 76.5% | 26/32 | [0.009787, 0.020394] |
| early-Q8-2 | 1.986 GiB | +182.8 MiB | 0.2055577 | 7.19% | 6.84% | 78.2% | 76.9% | 29/32 | [0.009716, 0.022111] |
| Q4_K_XL | 2.712 GiB | +926.3 MiB | 0.0165486 | 92.53% | 92.01% | 93.6% | 92.3% | 32/32 | [0.181952, 0.227893] |

The most size-efficient tested hybrid is **early-Q4-2**, recovering
6.81% KL for 66.9 MiB.
The best absolute hybrid is **early-Q8-2** at 7.19%
recovery. Raw Q2 KL is 0.2214713.

The Q4 donor retains 94.8% of the Q8 hybrid's KL reduction while using 36.6% as many
added bytes: 66.9 MiB (3.62% over Q2) rather than 182.8 MiB. Its chunk interval remains
above zero on prose. Five-repeat CUDA0 benchmarking gives 221.46 generation tok/s for the
Q4 donor and 221.64 for the Q8 donor, effectively identical and about 3.8% below the paired
Q2 baseline. Donor precision therefore changes memory much more than speed for these two
blocks.

Aggregate values are in `hybrid_precision.csv`; per-chunk metrics are in
`hybrid_precision_chunks.csv`. Intervals are descriptive t intervals over 32 chunks
from one prose corpus.
