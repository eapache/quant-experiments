# Next steps

## Scope of the current evidence

The completed real-model experiments test both the low-complexity end of the proposed
quantization-compensation hierarchy and one higher-capacity residual model:

- global temperature and top-p tuning;
- quant-logit-gap bucket corrections;
- a shrunken static per-token logit bias;
- entropy-conditioned token biases as a crude context-adaptive correction.
- a nested, head-restricted low-rank denoiser with oracle and quant-only amplitudes.

Deployable methods recovered only a few percent of divergence toward BF16. The low-rank
oracle recovered much more, but its quant-only predictor did not. The following ideas
remain untested on real GGUF logits.

## Completed: predictive low-rank logit denoiser

Fit a real-model version of

```text
corrected(q) = scale * q + token_bias + sum_j predictor(q)_j * residual_basis_j
```

Protocol:

1. Fit temperature and the regularized token bias on calibration chunks.
2. Form paired reference-minus-corrected-quant residuals.
3. Learn residual directions using weighted or head-restricted SVD. Weighting matters:
   the clipped vocabulary tail must not dominate ordinary Euclidean PCA.
4. Train a small ridge predictor for direction amplitudes using quant-only features such
   as entropy, top margins, head-logit projections, and candidate distribution shape.
5. At validation time, never use reference logits to choose amplitudes. Report an oracle
   projection separately only as an upper bound.
6. Select the number of directions and regularization through inner chunk-level
   validation, then report performance on untouched outer chunks. Proceed to the code
   corpus only if the predicted model beats static token bias.

This experiment is now implemented in `analyze_low_rank.py` and evaluated against BF16.
The key diagnostic gap is:

- oracle residual projection;
- quant-only predicted projection;
- static token bias.

The 16-direction oracle recovers 57.09% of Q2 KL and raises top-1 agreement from 77.3% to
94.4%. The nested quant-only predictor recovers only 1.44%, below the 1.65% static-bias
baseline; inner validation selects zero directions in three of four folds. Useful residual
structure exists, but its amplitudes are not inferable from the tested entropy, margin,
mass, and basis-projection features. See `results/bf16_low_rank_q2/LOW_RANK.md`.

## Priority 2: cumulative-mass calibration

The completed top-p experiment tuned one global candidate cutoff. It did **not** learn a
context-dependent mapping between candidate and reference cumulative mass.

For each calibration position:

1. Sort tokens by the quantized logits.
2. Measure how much BF16 probability mass is covered as quantized cumulative mass grows.
3. Fit a small monotonic mapping conditioned on quant-only statistics, for example:

```text
estimated_reference_mass = f(quant_mass, entropy, top_margin, head_shape)
```

4. At inference, retain the quantized prefix whose estimated reference mass reaches the
   user's requested top-p.

Compare against same top-p, globally tuned top-p, tuned temperature, and jointly tuned
temperature/top-p. Validate support overlap and JS divergence, not only the learned
mass-regression error.

## Priority 3: uncertainty-aware posterior sampler

Treat quantized candidate gaps as noisy observations of latent BF16 gaps. Estimate error
distributions from paired calibration data as a function of quant-only state:

```text
error scale = g(quant gap, rank, entropy, top margin, token identity/features)
```

Then approximate the posterior-predictive reference distribution over the plausible head:

```text
P(token | quant logits) = E[softmax(latent reference logits) | quant logits]
```

Start with the top 32-128 candidates and simple Gaussian or empirical residual models.
This must be evaluated against a deterministic mean-gap calibrator to show that modeling
uncertainty itself adds value, rather than merely applying another nonlinear logit map.

## Priority 4: nonlinear and pairwise gap calibration

The tested gap buckets were a coarse additive correction and did not help. Stronger
variants remain open:

- monotonic spline mapping from quant gap to estimated reference gap;
- rank-conditioned splines;
- entropy- or margin-conditioned splines;
- pairwise-gap regression among the plausible top tokens followed by consistent-logit
  reconstruction.

Constrain mappings to remain monotonic unless held-out evidence clearly supports rank
reversals. Otherwise a flexible fit can manufacture unstable ordering changes.

## Priority 5: token-feature correction

The static vocabulary bias used only paired output residuals. A more transferable token
correction could use properties of the quantized output row or token:

- output-embedding norm;
- row-level quantization error and outlier statistics;
- token frequency and category;
- punctuation, whitespace, code, and control-token indicators.

This requires access to corresponding higher-precision weights or quantization metadata,
not only saved output distributions. It is lower priority because it is more invasive and
less sampler-only.

## Experimental requirements

All advanced experiments should retain the safeguards established by the current work:

- Use identical teacher-forced tokens for reference and quant.
- Split and bootstrap by complete context chunk, never by adjacent prediction position.
- Tie artifacts to exact model hashes, llama.cpp commit, backend, prompt template, and
  reference sampler.
- Keep Q8-vs-quant claims distinct from BF16-vs-quant claims.
- Tune capacity and regularization only inside the training side of an outer split.
- Evaluate frozen settings or transforms on the independent Python-code corpus.
- Report raw divergence, held-out divergence, top-1/top-k agreement, TV/acceptance, and
  sampler-transformed JS/support overlap.
- Compare every complex model to temperature, global top-p, and static token-bias
  baselines. Complexity is justified only by repeatable held-out improvement.

## Recommended next decisive experiment

Test the uncertainty-aware posterior sampler over the top 32-128 Q2 candidates. The
low-rank oracle shows that head errors contain substantial correctable structure, while
the deterministic feature-to-amplitude predictor fails. Estimate rank-, gap-, and
entropy-conditioned residual distributions on inner calibration chunks, then compare a
posterior-predictive softmax against a deterministic conditional-mean correction on
untouched outer chunks. This directly tests whether acknowledging uncertain rank flips is
more useful than trying to guess one exact residual vector.
