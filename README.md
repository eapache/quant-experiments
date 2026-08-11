# Quant sampler compensation experiments

This repository tests whether paired next-token distributions can predict better sampling
settings, or learn a cheap pre-sampler correction, for a quantized language model.

## Bottom line

The simple sampler compensation idea works, but only weakly for the tested Qwen3.5-4B
GGUFs. The latest round uses the original BF16 GGUF as the reference and four-fold held-out
evaluation over 2,016 teacher-forced positions:

- Q8_K_XL is already close to BF16: raw KL is 0.00100 at T=1, sampler JS is 0.000873 at
  T=0.8/top-p=0.95, and the grid selects exactly the same settings in every fold.
- Q2_K_XL benefits from changing the BF16 sampler's T=0.8/top-p=0.95 to approximately
  **T=0.7625/top-p=0.95**. This recovers 2.22% of sampler JS. Joint T/top-p tuning recovers
  2.65%, but remains less identifiable.
- With those settings frozen on a Python-code corpus, Q2 temperature-only recovers 0.79%
  and joint tuning recovers 1.26%. Both per-chunk 95% intervals exclude zero. Top-p alone
  reverses sign.
- Q4_K_XL again wants effectively the original settings. Its frozen code-corpus gains are
  small and statistically inconclusive.
- At T=1, a shrunken static token bias recovers 2.88% of Q4 KL and 1.65% of Q2 KL toward
  BF16. It slightly worsens the already-small Q8 residual.
- The apparent top-128 low-rank oracle gap is mostly an oracle-capacity effect. Sixteen
  learned residual directions recover 57.09% of KL, but matched Gaussian random directions
  already recover 55.30%. Learned PCA contributes only 0.00395 additional KL reduction
  (3.1% of the total oracle reduction), with a four-block interval crossing zero. Singular
  values suggest some prose-specific covariance, but it is not portable: frozen on code,
  learned PCA loses to Gaussian directions at every rank and wins only 1/32 chunks at rank
  16. The data do not establish an intrinsically low-dimensional residual.
- A conditional empirical posterior over similar quant-logit heads is rejected in all
  four folds. Forced posterior prediction is safer than a conditional mean, but still
  recovers -0.53% of KL; the forced mean recovers -8.47%.
- A 1,624-6,448 parameter nonlinear predictor using the top-64 shape and residual-basis
  projections is also rejected in every fold. Forced use recovers -6.75% of KL while the
  same 16-direction oracle remains at 57.09%.
- A ridge map from all 2,560 dimensions of Q2's normalized final hidden state is rejected
  in every fold. Forced use recovers -4.44% of KL, while the corresponding oracle still
  recovers 57.01%. A larger paired corpus is required before attempting an in-model adapter.
- Replacing the tied Q6_K output head with BF16 weights only for the candidate's top tokens
  recovers 0.57% of Q2 KL in nested held-out evaluation. All four folds improve, but the
  correction has only 0.037 correlation with the full residual and a complete BF16 sidecar
  would cost about 1.18 GiB. Frozen on code at natural strength, recovery falls to 0.15%
  with a chunk interval crossing zero. Most degradation is upstream of the already-high-bit
  head.
- Residual-stream localization finds that Q2 error is created early: relative state error
  rises from 1.79% at the tied Q6_K embedding to 11.33% after the first recurrent block and
  22.14% before the first full-attention block. It peaks at 42.35% near layer 17. A global
  scale barely changes the error, ruling out simple hidden-state magnitude correction.
- A 10 KiB mean residual control vector before the first full-attention block recovers
  9.15% of held-out state MSE there, but only 0.02% of final KL. It improves 2/4 folds;
  the interval crosses zero. Frozen on code, it recovers 0.21% with only 19/32 chunks
  improved and another interval crossing zero. Correcting average hidden-state geometry is
  therefore not a sufficient proxy for correcting output distributions.
- Replacing just the first two complete Q2 recurrent blocks with byte-identical Q8 donor
  tensors recovers 7.19% of BF16-relative prose KL across 29/32 chunks. The hybrid adds
  182.8 MiB (9.88%) and reduces generation throughput by 4.59%. Frozen on code, it recovers
  5.41% across 27/32 chunks with a descriptive interval above zero. A third block has
  sharply worse marginal size efficiency, so the two-block design remains preferred.
- Using Q4 rather than Q8 for those same two donor blocks retains 94.8% of the prose KL
  reduction for only 66.9 MiB (3.62%) over Q2. It recovers 4.09% on code versus Q8's 5.41%,
  but its code chunk interval narrowly crosses zero. Q4 and Q8 donor generation speed is
  effectively identical, making Q4 the memory Pareto point and Q8 the better-supported one.
- Within the two Q4 blocks, the six FFN matrices recover 4.95% of prose KL for 38.7 MiB,
  carrying 72.7% of the full Q4 hybrid's reduction. The recurrent/SSM family recovers only
  1.91% for 28.3 MiB and has an interval crossing zero. Frozen on code, FFN-only recovers
  4.75% across 28/32 chunks and beats the complete Q4-block hybrid while using fewer bytes.
- Splitting those FFNs shows that up-only recovers 2.96% of prose KL for 10.5 MiB and
  gate-only recovers 2.34% for another 10.5 MiB. Down-only worsens KL by 0.49%. A predeclared
  gate+up combination recovers 5.02% for 21.1 MiB, slightly beating all-FFN. Gate+up is
  frozen as the primary sub-25-MiB design. It recovers 3.03% on code with an interval above
  zero and has no clearly measurable throughput penalty; up-only falls to 1.28% on code.
- Splitting gate+up by layer gives 2.81% prose recovery from block 1 alone for 10.5 MiB,
  versus 2.31% from block 0. Block 1 retains 56.0% of the two-block reduction for half the
  bytes and is frozen for code validation.
- Cumulative-mass calibration learns a stable global Q2 top-p near 0.939 and recovers
  0.64% sampler JS in held-out prose. Frozen on code, the gain falls to 0.06% with a 95%
  interval crossing zero. Temperature remains the strongest simple portable adjustment.
- Monotonic pairwise gap calibration edges the prose static-bias baseline by only 0.000101
  KL (1.70% versus 1.65% recovery), with a four-block interval crossing zero and unstable
  selected capacity. Frozen on code, its incremental gain is 0.000051 with a per-chunk
  interval crossing zero. The prose-trained static token bias itself recovers -4.72% on
  code, so neither vocabulary bias nor gap calibration is portable here.

The artifacts are tied to the exact model files, llama.cpp revision, CUDA backend, prompt
distribution, and reference sampler.

## Durable results

- [`NEXT_STEPS.md`](NEXT_STEPS.md): completed low-rank diagnostic and next quant-aware samplers
- [`results/bf16_gpu32/REPORT.md`](results/bf16_gpu32/REPORT.md): BF16-relative correction curves
- [`results/bf16_gpu32/SAMPLER.md`](results/bf16_gpu32/SAMPLER.md): BF16-relative sampler search
- [`results/bf16_low_rank_q2/LOW_RANK.md`](results/bf16_low_rank_q2/LOW_RANK.md): nested low-rank diagnostic
- [`results/bf16_low_rank_structure_q2/LOW_RANK_STRUCTURE.md`](results/bf16_low_rank_structure_q2/LOW_RANK_STRUCTURE.md): matched-null low-rank validation
- [`results/bf16_low_rank_structure_q2/FROZEN_LOW_RANK_STRUCTURE.md`](results/bf16_low_rank_structure_q2/FROZEN_LOW_RANK_STRUCTURE.md): frozen prose-to-code structure check
- [`results/bf16_posterior_q2/POSTERIOR.md`](results/bf16_posterior_q2/POSTERIOR.md): empirical posterior sampler
- [`results/bf16_neural_low_rank_q2/NEURAL_LOW_RANK.md`](results/bf16_neural_low_rank_q2/NEURAL_LOW_RANK.md): tiny nonlinear predictor
- [`results/bf16_hidden_q2/HIDDEN_ADAPTER.md`](results/bf16_hidden_q2/HIDDEN_ADAPTER.md): final-hidden-state ridge map
- [`results/bf16_output_head_q2/OUTPUT_HEAD.md`](results/bf16_output_head_q2/OUTPUT_HEAD.md): sparse BF16 output-head sidecar
- [`results/bf16_output_head_q2/FROZEN_OUTPUT_HEAD.md`](results/bf16_output_head_q2/FROZEN_OUTPUT_HEAD.md): frozen code-corpus head check
- [`results/bf16_layer_drift_q2/LAYER_DRIFT.md`](results/bf16_layer_drift_q2/LAYER_DRIFT.md): layerwise residual-stream localization
- [`results/bf16_mean_control_q2/MEAN_CONTROL.md`](results/bf16_mean_control_q2/MEAN_CONTROL.md): 10 KiB residual control-vector test
- [`results/bf16_mean_control_q2/FROZEN_MEAN_CONTROL.md`](results/bf16_mean_control_q2/FROZEN_MEAN_CONTROL.md): frozen code-corpus control check
- [`results/bf16_hybrid_precision/HYBRID_PRECISION.md`](results/bf16_hybrid_precision/HYBRID_PRECISION.md): selective early-block precision/size/speed tradeoff
- [`results/bf16_hybrid_precision/FROZEN_HYBRID_PRECISION.md`](results/bf16_hybrid_precision/FROZEN_HYBRID_PRECISION.md): frozen code-corpus hybrid check
- [`results/bf16_donor_precision/HYBRID_PRECISION.md`](results/bf16_donor_precision/HYBRID_PRECISION.md): Q4-versus-Q8 donor precision tradeoff
- [`results/bf16_tensor_families/HYBRID_PRECISION.md`](results/bf16_tensor_families/HYBRID_PRECISION.md): early-block FFN-versus-SSM ablation
- [`results/bf16_tensor_families/FROZEN_HYBRID_PRECISION.md`](results/bf16_tensor_families/FROZEN_HYBRID_PRECISION.md): frozen FFN-family code check
- [`results/bf16_ffn_matrices/HYBRID_PRECISION.md`](results/bf16_ffn_matrices/HYBRID_PRECISION.md): early FFN matrix ablation
- [`results/bf16_ffn_gate_up/HYBRID_PRECISION.md`](results/bf16_ffn_gate_up/HYBRID_PRECISION.md): predeclared gate+up combination
- [`results/bf16_ffn_gate_up/FROZEN_HYBRID_PRECISION.md`](results/bf16_ffn_gate_up/FROZEN_HYBRID_PRECISION.md): frozen compact gate+up code check
- [`results/bf16_gate_up_layers/HYBRID_PRECISION.md`](results/bf16_gate_up_layers/HYBRID_PRECISION.md): block-0-versus-block-1 gate+up split
- [`results/bf16_mass_q2/MASS_CALIBRATION.md`](results/bf16_mass_q2/MASS_CALIBRATION.md): cumulative-mass calibration
- [`results/bf16_mass_q2/FROZEN_MASS.md`](results/bf16_mass_q2/FROZEN_MASS.md): frozen code-corpus mass check
- [`results/bf16_gap_q2/GAP_CALIBRATION.md`](results/bf16_gap_q2/GAP_CALIBRATION.md): monotonic rank-conditioned gap calibration
- [`results/bf16_gap_q2/FROZEN_GAP.md`](results/bf16_gap_q2/FROZEN_GAP.md): frozen code-corpus gap check
- [`results/bf16_ood_code/SAMPLER.md`](results/bf16_ood_code/SAMPLER.md): frozen BF16-relative code check
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md): decisions, commands, environment, and findings
- [`results/gpu32/REPORT.md`](results/gpu32/REPORT.md): 63-1,512-position correction curves
- [`results/gpu32/SAMPLER.md`](results/gpu32/SAMPLER.md): held-out temperature/top-p search
- [`results/ood_code/SAMPLER.md`](results/ood_code/SAMPLER.md): frozen out-of-domain check
- [`results/REPORT.md`](results/REPORT.md): initial dense 504-position experiment
- [`results/LOW_SAMPLE.md`](results/LOW_SAMPLE.md): initial low-sample stability experiment

The large `.kld` captures are reproducible and intentionally ignored by Git. Compact
fold-level CSV results are versioned.

## Main workflow

llama.cpp's perplexity tool writes compact full-vocabulary log-probabilities when
`--kl-divergence-base` is supplied without `--kl-divergence`. Generate matching Q8, Q4,
and Q2 captures with the command recorded in `EXPERIMENT_LOG.md`, then run:

```bash
python3 analyze_sparse.py \
  --reference results/logits/q8-32.kld \
  --q4 results/logits/q4-32.kld \
  --q2 results/logits/q2-32.kld \
  --output-dir results/gpu32

python3 analyze_sampler_grid.py \
  --reference results/logits/q8-32.kld \
  --q4 results/logits/q4-32.kld \
  --q2 results/logits/q2-32.kld \
  --output-dir results/gpu32

python3 evaluate_frozen_sampler.py \
  --reference results/logits/q8-code32.kld \
  --q4 results/logits/q4-code32.kld \
  --q2 results/logits/q2-code32.kld \
  --output-dir results/ood_code
```

The BF16-relative commands and low-rank experiment are recorded in `EXPERIMENT_LOG.md`.

The final-hidden-state experiment additionally builds `extract_hidden_states.cpp` with
`./build_hidden_extractor.sh`, extracts CUDA0-aligned Q2 states from a `.kld` token header,
and runs `analyze_hidden_adapter.py`. Exact commands are in `EXPERIMENT_LOG.md`.

NumPy is the only non-standard Python dependency for the sparse analyses. The dense
analyzer also uses SciPy.
