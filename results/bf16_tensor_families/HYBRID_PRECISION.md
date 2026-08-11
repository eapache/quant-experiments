# Selective early-block precision

The hybrids retain every tensor byte from Q2_K_XL except selected tensors from early
recurrent blocks copied from a shape-identical higher-precision GGUF. The builder reopens
each model and verifies every raw tensor payload against its intended source. This
experiment was chosen from residual-state localization without inspecting hybrid output
logits.

| Model | Size | Added | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 baseline | 1.808 GiB | +0.0 MiB | 0.2214713 | 0.00% | 0.00% | 77.3% | 75.7% | — | — |
| Q4-SSM-only | 1.835 GiB | +28.3 MiB | 0.2172442 | 1.91% | 1.58% | 77.0% | 76.0% | 21/32 | [-0.000224, 0.008678] |
| Q4-FFN-only | 1.845 GiB | +38.7 MiB | 0.2105060 | 4.95% | 4.66% | 79.0% | 76.5% | 28/32 | [0.005767, 0.016164] |
| Q4-full-blocks | 1.873 GiB | +66.9 MiB | 0.2063805 | 6.81% | 6.29% | 78.7% | 76.5% | 26/32 | [0.009787, 0.020394] |
| Q8-full-blocks | 1.986 GiB | +182.8 MiB | 0.2055577 | 7.19% | 6.84% | 78.2% | 76.9% | 29/32 | [0.009716, 0.022111] |

The most size-efficient tested hybrid is **Q4-FFN-only**, recovering
4.95% KL for 38.7 MiB.
The best absolute hybrid is **Q4-full-blocks** at 6.81%
recovery. Raw Q2 KL is 0.2214713.

The FFN-only upgrade carries 72.7% of the complete Q4 hybrid's absolute KL reduction with
57.8% of its added bytes. Its KL reduction per added MiB is 1.90 times the SSM-only
variant's, it improves 28/32 chunks, and its interval stays above zero. The SSM-only
interval crosses zero. The two effects are nearly additive in aggregate, but FFN-only is
the clear family to freeze on code before considering individual matrices.

Marginal KL reduction per added MiB between successive listed hybrids is 0.00014957,
0.00064728, and 0.00014597, respectively. Because the first two variants are disjoint
rather than nested, their direct baseline-relative efficiencies are the meaningful
comparison; those are 0.00014957 for SSM and 0.00028354 for FFN.

Aggregate values are in `hybrid_precision.csv`; per-chunk metrics are in
`hybrid_precision_chunks.csv`. Intervals are descriptive t intervals over 32 chunks
from one prose corpus.
