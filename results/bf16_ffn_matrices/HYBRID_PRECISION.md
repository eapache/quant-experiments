# Selective early-block precision

The hybrids retain every tensor byte from Q2_K_XL except selected tensors from early
recurrent blocks copied from a shape-identical higher-precision GGUF. The builder reopens
each model and verifies every raw tensor payload against its intended source. This experiment was
chosen from residual-state localization without inspecting hybrid output logits.

| Model | Size | Added | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | 1.808 GiB | +0.0 MiB | 0.2214713 | 0.00% | 0.00% | 77.3% | 75.7% | — | — |
| Q4-gate-only | 1.818 GiB | +10.5 MiB | 0.2162851 | 2.34% | 2.34% | 78.4% | 76.2% | 23/32 | [0.001779, 0.008593] |
| Q4-up-only | 1.818 GiB | +10.5 MiB | 0.2149065 | 2.96% | 2.65% | 77.9% | 76.2% | 24/32 | [0.002439, 0.010691] |
| Q4-down-only | 1.825 GiB | +17.6 MiB | 0.2225600 | -0.49% | -0.37% | 77.3% | 75.9% | 13/32 | [-0.003163, 0.000985] |
| Q4-all-FFN | 1.845 GiB | +38.7 MiB | 0.2105060 | 4.95% | 4.66% | 79.0% | 76.5% | 28/32 | [0.005767, 0.016164] |
| Q4-full-blocks | 1.873 GiB | +66.9 MiB | 0.2063805 | 6.81% | 6.29% | 78.7% | 76.5% | 26/32 | [0.009787, 0.020394] |

The most size-efficient tested hybrid is **Q4-up-only**, recovering
2.96% KL for 10.5 MiB.
The best absolute hybrid is **Q4-all-FFN** at 4.95%
recovery. Raw Q2 KL is 0.2214713.

Baseline-relative KL reduction per added MiB is Q4-gate-only=0.00049172, Q4-up-only=0.00062243, Q4-down-only=-0.00006194, Q4-all-FFN=0.00028355.

Up-only is the best tested memory point: its two matrices recover 2.96% for 10.5 MiB,
improve 24/32 chunks, and have an interval above zero. Gate-only is also positive at 2.34%
for the same size. Down-only reverses sign, worsening KL by 0.49% and improving only 13/32
chunks. The gate and up reductions sum to slightly more than the complete FFN reduction,
so the next predeclared test is their 21.1 MiB combination without the harmful down matrix.
These matrix choices are exploratory on the same prose corpus and need frozen code checks.

Aggregate values are in `hybrid_precision.csv`; per-chunk metrics are in
`hybrid_precision_chunks.csv`. Intervals are descriptive t intervals over 32 chunks
from one prose corpus.
