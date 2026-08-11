# Multi-document compact-hybrid confirmation

**Q4 gate+up blocks 0-1** was fixed before any captures from these documents were
generated. Every predeclared document is retained below.

| Document | Positions | Q2 KL | Candidate KL | KL recovered | JS recovered | Better chunks | Chunk interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| real-logits | 2016 | 0.1789505 | 0.1703725 | 4.79% | 4.13% | 25/32 | [0.001408, 0.015748] |
| low-rank-controls | 2016 | 0.1790127 | 0.1719623 | 3.94% | 3.97% | 26/32 | [0.002166, 0.011935] |
| gap-calibration | 2016 | 0.1709858 | 0.1677928 | 1.87% | 1.47% | 19/32 | [-0.000713, 0.007099] |
| hidden-adapter | 2016 | 0.1773713 | 0.1678169 | 5.39% | 3.92% | 24/32 | [-0.001874, 0.020983] |

The candidate improves 4/4 whole
documents. Pooled over 8064 positions, KL changes from
0.1765801 to 0.1694861,
recovering 4.02%; pooled JS recovery is
3.38%.

Mean whole-document KL reduction is 0.0070940
with descriptive four-document t interval [0.0026433,
0.0115446]. This is the primary uncertainty summary;
chunk intervals describe within-file consistency only.

The documents are separate implementations from one repository and one language, not
independent samples of all deployment workloads. Aggregate values are in
`document_confirmation.csv`; chunk values are in `document_confirmation_chunks.csv`.
