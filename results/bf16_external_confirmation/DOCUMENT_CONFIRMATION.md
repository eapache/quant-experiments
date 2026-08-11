# Multi-document compact-hybrid confirmation

**Q4 gate+up blocks 0-1** was fixed before any captures from these documents were
generated. Every predeclared document is retained below.

| Document | Positions | Q2 KL | Candidate KL | KL recovered | JS recovered | Better chunks | Chunk interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| stdlib-argparse | 2016 | 0.2258741 | 0.2205614 | 2.35% | 2.58% | 21/32 | [-0.000415, 0.011040] |
| llama-sampling | 2016 | 0.1840590 | 0.1798454 | 2.29% | 2.03% | 22/32 | [-0.002077, 0.010504] |
| server-docs | 2016 | 0.2492775 | 0.2299602 | 7.75% | 5.32% | 24/32 | [0.007758, 0.030877] |
| gpl3 | 2016 | 1.2777417 | 1.2549806 | 1.78% | 1.83% | 22/32 | [0.005986, 0.039536] |

The candidate improves 4/4 whole
documents. Pooled over 8064 positions, KL changes from
0.4842381 to 0.4713369,
recovering 2.66%; pooled JS recovery is
2.46%.

Mean whole-document KL reduction is 0.0129012
with descriptive four-document t interval [-0.0022326,
0.0280350]. This is the primary uncertainty summary;
chunk intervals describe within-file consistency only.

The documents are a small fixed set, not independent samples of all deployment
workloads. Aggregate values are in
`document_confirmation.csv`; chunk values are in `document_confirmation_chunks.csv`.
