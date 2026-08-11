# Mean residual-stream control vector

Each outer fold fits one 2,560-float vector: the mean BF16-minus-Q2 input-state
difference at layer 3, using only the other 24 context chunks. llama.cpp
adds it after block 2, immediately before the measured target state.
No BF16 output logits or held-out states are used to fit the sidecar, and strength is
fixed at its natural value of 1.

| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw Q2 | 0.2214713 | 0.00% | 0.0510455 | 0.2140513 | 77.3% | 75.7% |
| Mean layer control | 0.2214191 | 0.02% | 0.0509932 | 0.2134115 | 77.6% | 76.0% |

The control vector improves 2/4 outer folds.
Mean KL reduction is 0.0000522; its four-block t interval is
[-0.0013435, 0.0014478].

Fold metrics are in `mean_control_folds.csv`; vector metadata is in
`control_vectors.csv`.
