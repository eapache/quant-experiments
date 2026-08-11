# Frozen hybrid temperature across documents

The compact hybrid and T=0.7625/top-p=0.95 were fixed
before this evaluation. Every available out-of-selection document is retained.

| Document | Q2 JS | Hybrid JS | +temperature JS | Model recovery | Temperature recovery | Total recovery | Better chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-lab | 0.0512854 | 0.0493727 | 0.0488957 | 3.73% | 0.97% | 4.66% | 25/32 |
| real-logits | 0.0495104 | 0.0478128 | 0.0472684 | 3.43% | 1.14% | 4.53% | 21/32 |
| low-rank-controls | 0.0504955 | 0.0484004 | 0.0475787 | 4.15% | 1.70% | 5.78% | 28/32 |
| gap-calibration | 0.0487584 | 0.0481677 | 0.0472547 | 1.21% | 1.90% | 3.08% | 29/32 |
| hidden-adapter | 0.0490578 | 0.0472617 | 0.0466783 | 3.66% | 1.23% | 4.85% | 24/32 |
| stdlib-argparse | 0.0591372 | 0.0578246 | 0.0571261 | 2.22% | 1.21% | 3.40% | 25/32 |
| llama-sampling | 0.0475668 | 0.0468414 | 0.0461150 | 1.53% | 1.55% | 3.05% | 26/32 |
| server-docs | 0.0625767 | 0.0594441 | 0.0585820 | 5.01% | 1.45% | 6.38% | 26/32 |
| gpl3 | 0.2192950 | 0.2149901 | 0.2140199 | 1.96% | 0.45% | 2.41% | 28/32 |

Temperature compensation improves 9/
9 whole documents. Across 18144 positions,
pooled hybrid JS falls from 0.0689017 to
0.0681687, recovering
1.06%.
Raw Q2 pooled JS is 0.0708537; the model alone recovers
2.75%, and model plus temperature recovers
3.79% in total.

Mean whole-document JS reduction is 0.0007330
with descriptive document interval [0.0006004,
0.0008655]. Document variation is the primary
uncertainty summary; chunks are correlated positions within files.

Aggregate results are in `frozen_hybrid_sampler_documents.csv`; chunk results are in
`frozen_hybrid_sampler_document_chunks.csv`.
