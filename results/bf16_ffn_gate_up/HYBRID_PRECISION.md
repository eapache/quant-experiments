# Selective early-block precision

The hybrids retain every tensor byte from Q2_K_XL except selected tensors from early
recurrent blocks copied from a shape-identical higher-precision GGUF. The builder reopens
each model and verifies every raw tensor payload against its intended source. This experiment was
chosen from residual-state localization without inspecting hybrid output logits.

| Model | Size | Added | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | 1.808 GiB | +0.0 MiB | 0.2214713 | 0.00% | 0.00% | 77.3% | 75.7% | — | — |
| Q4-up-only | 1.818 GiB | +10.5 MiB | 0.2149065 | 2.96% | 2.65% | 77.9% | 76.2% | 24/32 | [0.002439, 0.010691] |
| Q4-gate-only | 1.818 GiB | +10.5 MiB | 0.2162851 | 2.34% | 2.34% | 78.4% | 76.2% | 23/32 | [0.001779, 0.008593] |
| Q4-gate+up | 1.828 GiB | +21.1 MiB | 0.2103633 | 5.02% | 4.52% | 78.7% | 76.4% | 27/32 | [0.005122, 0.017094] |
| Q4-all-FFN | 1.845 GiB | +38.7 MiB | 0.2105060 | 4.95% | 4.66% | 79.0% | 76.5% | 28/32 | [0.005767, 0.016164] |

The most size-efficient tested hybrid is **Q4-up-only**, recovering
2.96% KL for 10.5 MiB.
The best absolute hybrid is **Q4-gate+up** at 5.02%
recovery. Raw Q2 KL is 0.2214713.

Baseline-relative KL reduction per added MiB is Q4-up-only=0.00062243, Q4-gate-only=0.00049172, Q4-gate+up=0.00052659, Q4-all-FFN=0.00028355.

The predeclared gate+up combination works: it recovers 5.02% for 21.1 MiB, improves 27/32
chunks, and has an interval above zero. It slightly beats all-FFN's 4.95% while removing
the 17.6 MiB down matrices, confirming that the down upgrade is unnecessary here. Gate+up
is frozen as the primary sub-25-MiB design for code. Up-only remains a secondary 10.5 MiB
Pareto endpoint, not an alternative selected from code.

Aggregate values are in `hybrid_precision.csv`; per-chunk metrics are in
`hybrid_precision_chunks.csv`. Intervals are descriptive t intervals over 32 chunks
from one prose corpus.
