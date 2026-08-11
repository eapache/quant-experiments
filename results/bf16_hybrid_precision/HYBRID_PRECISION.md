# Selective early-block precision

The hybrids retain every tensor byte from Q2_K_XL except complete early recurrent
blocks copied from a shape-identical higher-precision GGUF. The builder reopens each model
and verifies every raw tensor payload against its intended source. This experiment was
chosen from residual-state localization without inspecting hybrid output logits.

| Model | Size | Added | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | 1.808 GiB | +0.0 MiB | 0.2214713 | 0.00% | 0.00% | 77.3% | 75.7% | — | — |
| early-Q8-1 | 1.897 GiB | +91.4 MiB | 0.2141036 | 3.33% | 3.03% | 77.8% | 76.1% | 26/32 | [0.002951, 0.011784] |
| early-Q8-2 | 1.986 GiB | +182.8 MiB | 0.2055577 | 7.19% | 6.84% | 78.2% | 76.9% | 29/32 | [0.009716, 0.022111] |
| early-Q8-3 | 2.130 GiB | +330.7 MiB | 0.2019147 | 8.83% | 8.31% | 78.8% | 77.0% | 30/32 | [0.014128, 0.024985] |
| Q4_K_XL | 2.712 GiB | +926.3 MiB | 0.0165486 | 92.53% | 92.01% | 93.6% | 92.3% | 32/32 | [0.181952, 0.227893] |

The most size-efficient tested hybrid is **early-Q8-2**, recovering
7.19% KL for 182.8 MiB.
The best absolute hybrid is **early-Q8-3** at 8.83%
recovery. Raw Q2 KL is 0.2214713.

Marginal KL reduction per added MiB for blocks 1, 2, and 3 is 0.00008060, 0.00009348, 0.00002464, respectively.

Generation throughput makes the same two-block model the practical choice. Five repeated
CUDA0 measurements at prompt 128 / generation 32 give:

| Model | Prompt tok/s | Change | Generation tok/s | Change |
|---|---:|---:|---:|---:|
| Q2 baseline | 5951.2 | — | 233.82 | — |
| early-Q8-1 | 5865.8 | -1.43% | 227.89 | -2.54% |
| early-Q8-2 | 6022.7 | +1.20% | 223.09 | -4.59% |
| early-Q8-3 | 6023.4 | +1.21% | 214.32 | -8.34% |
| Q4_K_XL | 6976.9 | +17.24% | 193.09 | -17.42% |

Prompt-processing differences among the Q2 hybrids are small relative to run-to-run
variation and should be treated as unchanged. Generation slows monotonically as Q8 donor
blocks are added. **early-Q8-2 is the frozen design:** it recovers 7.19% of BF16-relative
KL for 182.8 MiB (9.88% over the Q2 file) and a 4.59% generation slowdown. Adding block 2
raises overhead to 330.7 MiB and slowdown to 8.34%, but yields only another 1.64 percentage
points of recovery because that donor block contains expensive F16 tensors. Q4 remains
far more accurate, but costs 926.3 MiB and 17.42% generation throughput relative to Q2.

Aggregate values are in `hybrid_precision.csv`; per-chunk metrics are in
`hybrid_precision_chunks.csv`; exact model hashes and benchmark results are in
`hybrid_models.csv` and `hybrid_throughput.csv`. Intervals are descriptive t intervals
over 32 chunks from one prose corpus.
