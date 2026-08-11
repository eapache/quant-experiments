# Selective early-block precision

The hybrids retain every tensor byte from Q2_K_XL except selected tensors from early
recurrent blocks copied from a shape-identical higher-precision GGUF. The builder reopens
each model and verifies every raw tensor payload against its intended source. This experiment was
chosen from residual-state localization without inspecting hybrid output logits.

| Model | Size | Added | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | 1.808 GiB | +0.0 MiB | 0.2214713 | 0.00% | 0.00% | 77.3% | 75.7% | — | — |
| Q4-block0-gate+up | 1.818 GiB | +10.5 MiB | 0.2163606 | 2.31% | 2.13% | 78.0% | 76.2% | 23/32 | [0.001954, 0.008267] |
| Q4-block1-gate+up | 1.818 GiB | +10.5 MiB | 0.2152561 | 2.81% | 2.50% | 78.2% | 76.0% | 24/32 | [0.002977, 0.009454] |
| Q4-blocks0+1-gate+up | 1.828 GiB | +21.1 MiB | 0.2103633 | 5.02% | 4.52% | 78.7% | 76.4% | 27/32 | [0.005122, 0.017094] |
| Q4-all-FFN | 1.845 GiB | +38.7 MiB | 0.2105060 | 4.95% | 4.66% | 79.0% | 76.5% | 28/32 | [0.005767, 0.016164] |

The most size-efficient tested hybrid is **Q4-block1-gate+up**, recovering
2.81% KL for 10.5 MiB.
The best absolute hybrid is **Q4-blocks0+1-gate+up** at 5.02%
recovery. Raw Q2 KL is 0.2214713.

Baseline-relative KL reduction per added MiB is Q4-block0-gate+up=0.00048456, Q4-block1-gate+up=0.00058928, Q4-blocks0+1-gate+up=0.00052659.

Block 1 is the better single-block point. It recovers 2.81% for 10.5 MiB, improves 24/32
chunks, and has an interval above zero; block 0 recovers 2.31% and improves 23/32. Their
reductions are approximately additive. Block 1 retains 56.0% of the two-block reduction
for half the bytes and is frozen as the single-block primary before code capture.

Aggregate values are in `hybrid_precision.csv`; per-chunk metrics are in
`hybrid_precision_chunks.csv`. Intervals are descriptive t intervals over 32 chunks
from one prose corpus.
