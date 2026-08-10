# Next steps

## Scope of the current evidence

The completed real-model experiments mainly tested the low-complexity end of the proposed
quantization-compensation hierarchy:

- global temperature and top-p tuning;
- quant-logit-gap bucket corrections;
- a shrunken static per-token logit bias;
- entropy-conditioned token biases as a crude context-adaptive correction.

These methods recovered only a few percent of divergence toward Q8. That result does
**not** rule out the more expressive methods discussed in `chats/first.txt`. The following
ideas remain untested on real GGUF logits.

## Priority 1: predictive low-rank logit denoiser

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
   validation, then report performance on untouched outer chunks and the code corpus.

The key diagnostic is the gap between:

- oracle residual projection;
- quant-only predicted projection;
- static token bias.

If the oracle projection is strong but prediction is weak, useful residual structure
exists but is not inferable at sampling time. If even the oracle is weak, low-rank output
directions are not a promising representation for this quant.

## Priority 2: cumulative-mass calibration

The completed top-p experiment tuned one global candidate cutoff. It did **not** learn a
context-dependent mapping between candidate and reference cumulative mass.

For each calibration position:

1. Sort tokens by the quantized logits.
2. Measure how much Q8 probability mass is covered as quantized cumulative mass grows.
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

Treat quantized candidate gaps as noisy observations of latent Q8 gaps. Estimate error
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

Implement the predictive low-rank denoiser first, using the existing 32-chunk CUDA
captures. It directly tests the main untried Heretic-like hypothesis without requiring
new inference. Begin with an oracle upper bound and probability-weighted SVD; proceed to
a quant-only amplitude predictor only if that upper bound materially exceeds the static
token-bias result.
