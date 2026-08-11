# Temperature compensation stacked on compact hybrid

The target is BF16 at T=0.80/top-p=0.95.
Four outer folds compare identical hybrid settings, the previously frozen Q2
temperature T=0.7625, and temperature refit only on other prose chunks.
A hybrid-specific setting is promoted only if it beats the Q2-frozen temperature
with a paired 32-chunk interval above zero.

| Method | JS | Recovery vs same settings | TV | Candidate T |
|---|---:|---:|---:|---:|
| same-settings | 0.0634714 | 0.00% | 0.2118071 | 0.8000 |
| q2-frozen-temperature | 0.0619946 | 2.33% | 0.2074608 | 0.7625 |
| hybrid-tuned-temperature | 0.0620121 | 2.30% | 0.2072624 | 0.7556 |

Nested hybrid temperatures are 0.7550+0.7675+0.7500+0.7500 (mean
0.7556). The hybrid-specific setting improves
15/32 chunks over Q2's frozen temperature.
Mean incremental JS reduction is -0.0000175,
with interval [-0.0001402,
0.0001051]. It **fails** the promotion gate.

The existing Q2 temperature therefore stacks cleanly with the hybrid and remains the
recommendation. It adds no model bytes or inference operation beyond changing the sampler
parameter. The hybrid-specific mean is not promoted.

Frozen results across nine out-of-selection documents are in
`FROZEN_HYBRID_SAMPLER.md`: temperature improves all 9/9 documents and adds 1.06% pooled
sampler-JS recovery. Model plus temperature recovers 3.79% from raw Q2 in total.

Fold results are in `hybrid_sampler.csv`; paired chunk results are in
`hybrid_sampler_chunks.csv`.
