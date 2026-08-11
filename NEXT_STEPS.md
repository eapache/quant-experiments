# Next steps

## Scope of the current evidence

The completed real-model experiments test the low-complexity end of the proposed
quantization-compensation hierarchy and several higher-capacity residual models:

- global temperature and top-p tuning;
- quant-logit-gap bucket corrections;
- a shrunken static per-token logit bias;
- entropy-conditioned token biases as a crude context-adaptive correction;
- a nested, head-restricted low-rank denoiser with oracle and quant-only amplitudes;
- global and entropy-conditioned cumulative-mass calibration;
- a conditional empirical posterior over correlated rank-wise head errors;
- a 1,624-6,448 parameter nonlinear predictor of low-rank residual amplitudes;
- monotonic direct and consistent pairwise rank-gap calibration;
- a 2,560-feature final-hidden-state ridge map to residual amplitudes.

Deployable sampler methods recovered only a few percent of divergence toward BF16. The
low-rank oracle recovered much more, although matched random directions explain nearly all
of that oracle recovery. Ridge, nearest-neighbor posterior, and tiny nonlinear quant-only
predictors did not improve. Selectively upgrading the first two recurrent blocks is the
first model-side intervention to recover a clear 7.19% of KL at modest cost.

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
baseline; inner validation selects zero directions in three of four folds. A later matched-
null diagnostic substantially changes the oracle interpretation, as described below. See
`results/bf16_low_rank_q2/LOW_RANK.md`.

## Completed: matched-null low-rank validation

`analyze_low_rank_structure.py` gives learned and control subspaces the same held-out BF16
oracle and the same number of per-row amplitudes. Controls include Gaussian random
subspaces, token-permuted learned directions, and PCA after independently randomizing all
training residual signs. All bases remain outer-fold clean.

At rank 16, learned residual PCA recovers 57.09% of raw KL, but Gaussian random directions
already recover 55.30%, sign-randomized PCA recovers 51.82%, and token-permuted PCA recovers
34.65%. The learned advantage over the mean Gaussian oracle is only 0.00395 KL, or 3.1% of
the learned oracle's total reduction. Its four-block t interval is
[-0.00043, 0.00834]. Learned singular values exceed the sign-null values, especially in the
first components, but learned and Gaussian rank curves track closely through rank 32 and
neither has a sharp low-dimensional cutoff.

The corrected conclusion is that coherent token-aligned covariance probably exists, but
the 57% oracle recovery is mostly the generic power of fitting 16 target-aware coefficients
to distributions whose median effective support is only about 10 tokens for BF16. It does
not prove a 16-dimensional residual or establish amplitude prediction as the sole missing
ingredient. See `results/bf16_low_rank_structure_q2/LOW_RANK_STRUCTURE.md`.

The prose-trained bases were then frozen and given oracle amplitudes on the existing code
capture. Gaussian directions beat learned PCA at every tested rank. At rank 16, learned
PCA reaches KL 0.07714 (56.96% recovery), while Gaussian directions reach 0.06668 (62.79%).
Learned PCA wins only 1/32 chunks; its mean advantage is -0.01046 with a descriptive
chunk-level t interval [-0.01377, -0.00714]. Although this reused single-file corpus is not
a pristine confirmation set, the large reversal is evidence against portable low-rank
residual structure. See
`results/bf16_low_rank_structure_q2/FROZEN_LOW_RANK_STRUCTURE.md`.

## Completed: cumulative-mass calibration

`analyze_mass_calibration.py` learns the monotonic mapping between quant prefix mass and
BF16 mass, either globally or in quant-entropy bins. The global mapping implies Q2 top-p
about 0.939 for the BF16 top-p=0.95 target and recovers 0.64% held-out prose JS. Entropy
conditioning is enabled in only one of four inner selections and slightly worsens the
aggregate result. Frozen on code, the global mapping recovers only 0.06%, with a per-chunk
95% interval crossing zero. This does not beat tuned temperature and mostly rediscovers
the workload-sensitive global top-p shift.

## Completed: uncertainty-aware empirical posterior sampler

`analyze_posterior.py` retrieves training rows with similar quant-only entropy, mass, and
top-gap shape. It compares a rank-wise conditional-mean residual with the average of
softmax distributions under correlated empirical residual samples. Head and neighbor
count are nested-selected, with a null option.

The posterior mixture is substantially safer than choosing the conditional mean: forced
use recovers -0.53% versus -8.47% KL. Both are rejected in all four folds and fall back to
the 1.65% static-token baseline. Modeling uncertainty in this simple empirical form does
not unlock the oracle residual structure.

## Completed: tiny nonlinear amplitude predictor

`analyze_neural_low_rank.py` is a deliberately small learned-model test before attempting
a transformer or LoRA. A one-hidden-layer tanh network receives the full top-64 quant gap
and probability shape plus candidate projections onto the residual basis, and predicts
the 16 oracle amplitudes. Nested selection covers 1,624-6,448 parameters, weight decay,
and training duration.

All four folds reject the network. Forced use recovers -6.75% KL and worsens top-10
overlap, while the 16-direction oracle still recovers 57.09%. The matched-null result shows
that this oracle contrast is mostly generic target-aware capacity, and coefficient MSE is
only a surrogate for KL. The result rejects this network and training target, not every
possible function of the saved output distribution. A full transformer remains an
inefficient next step without a substantially larger paired corpus.

## Completed: nonlinear and pairwise gap calibration

`analyze_gap_calibration.py` fits weighted monotonic piecewise-linear mappings from
candidate gaps to BF16 gaps after temperature and static-token correction. The direct
variant maps top-anchor gaps. The pairwise variant maps every plausible head-token pair
and reconstructs one consistent logit vector. Head size, knot count, rank conditioning,
and correction strength are nested-selected with a null option.

Direct calibration is harmful on held-out prose. Pairwise calibration moves KL from the
static-bias baseline's 0.2178015 to 0.2177002, or 1.70% rather than 1.65% recovery from
raw. The incremental 0.000101 gain has a four-block 95% interval crossing zero, while
selected head size and strength are unstable. A prose-fitted head-8 transform frozen on
code improves its static-bias baseline by only 0.000051 KL, interval
[-0.000190, 0.000292]. The static bias itself transfers poorly, worsening raw code KL by
4.72%. Gap calibration therefore does not provide repeatable or portable improvement.

## Completed first pass: final-hidden-state residual map

`extract_hidden_states.cpp` uses llama.cpp's staging API to save the normalized Q2 final
hidden state immediately before the LM head. It consumes the token header from the `.kld`
file, so all 2,016 rows align exactly with the existing CUDA0 BF16/Q2 distributions.
`analyze_hidden_adapter.py` fits a kernel-form ridge map from all 2,560 hidden dimensions
to the same head-restricted oracle residual amplitudes used by the low-rank diagnostic.
Direction count and ridge strength are nested-selected with a null option.

All four folds reject the hidden map. Forced use chooses very strong regularization and
worsens KL from the static-bias baseline's 0.2178012 to 0.2313056, or -4.44% recovery
from raw. The 16-direction oracle still recovers 57.01%. The normalized final state does
not predict these oracle residual amplitudes at this sample size; because it feeds a linear
output head, it is also closely related to information already present in the full logits.
This does not replace a direct-KL training test. See
`results/bf16_hidden_q2/HIDDEN_ADAPTER.md`.

## Completed: sparse high-precision output-head sidecar

`analyze_output_head.py` uses the captured Q2 final state and exact GGUF tensor rows to
apply `(W_BF16 - W_Q6) h` only to the quant candidate's top-N tokens. Head size and blend
strength are selected on an inner chunk split. Recomputed Q6 logits match the saved logits
to 0.027 centered RMSE over the top-256 sweep, validating the state/tensor alignment.

Nested selection recovers 0.57% of raw Q2 KL (0.2214713 to 0.2202139). Every outer fold
improves; the mean reduction is 0.0012574 with a four-block t interval
[0.0003679, 0.0021468]. Top-1 agreement rises from 77.3% to 77.8%, while top-10 overlap is
unchanged. The head correction's RMS is only 0.036 versus 0.959 for the full residual on
the same tokens, with correlation 0.037. This is direct evidence that most low-bit damage
is created upstream of the output head. The Q2 model already keeps its tied head at Q6_K,
and a complete BF16 sidecar would add about 1.18 GiB, so this is not competitive with the
smaller, similarly effective temperature adjustment. See
`results/bf16_output_head_q2/OUTPUT_HEAD.md`.

The natural-strength head-256 correction was then frozen before loading the code capture.
It recovers only 0.15% of code KL, improves 20/32 chunks, and has mean per-chunk reduction
0.0002729 with a descriptive interval [-0.0001954, 0.0007413]. Its correction/residual
correlation falls to 0.028. This is consistent with a tiny real head-quantization effect,
but it is neither large nor stable enough to justify a BF16 sidecar. See
`results/bf16_output_head_q2/FROZEN_OUTPUT_HEAD.md`.

## Completed: residual-stream drift localization

`extract_layer_states.cpp` uses llama.cpp's layer-input output hook to capture all 32
unnormalized residual-stream states on the exact teacher-forced positions. Comparing BF16
and Q2 with `analyze_layer_drift.py` shows that the error is created early and is strongly
directional rather than a scale mismatch. Relative state error is 1.79% at the Q6_K input
embedding, 11.33% after the first recurrent block, and 22.14% before the first
full-attention block. The first three recurrent blocks therefore create the steepest drift;
the first full-attention block does not add to relative error. Error peaks at 42.35% at
layer 17 and ends at 39.91% before the final block, while absolute error continues growing.

A single global candidate scale barely changes any layer's error. Held-out mean state bias
recovers 9.15% of state MSE at layer 3 but only about 3-5% through most later layers.
Per-row drift has essentially zero correlation with final KL early, rising to 0.27 at
layers 30-31. This points to a cheap targeted test: inject the prose-trained mean residual
immediately before the first full-attention block, then measure held-out logits and freeze
it on code. If that fails, selectively retaining the first recurrent blocks at higher
precision is more motivated than upgrading the output head. See
`results/bf16_layer_drift_q2/LAYER_DRIFT.md`.

## Completed: mean residual-stream control vector

`build_mean_control_vectors.py` produces outer-fold-clean 10,464-byte GGUF sidecars from
the mean BF16-minus-Q2 state difference at layer 3. llama.cpp adds each 2,560-float vector
after block 2 at natural strength 1. This uses no BF16 output logits, and fold vectors have
stable RMS near 0.00412.

Despite recovering 9.15% of held-out state MSE at the target layer, the vector reduces
final KL only from 0.2214713 to 0.2214191 (0.02%). It improves 2/4 outer folds; mean KL
reduction is 0.0000522 with interval [-0.0013435, 0.0014478]. TV and top-10 overlap improve
slightly, but average hidden-state MSE is not a useful proxy for BF16 KL. The next state-
based correction would need a KL-aware direction or context-dependent coefficients; a
larger static control vector is not justified. See
`results/bf16_mean_control_q2/MEAN_CONTROL.md`.

The all-prose vector was frozen on the independent code capture. It reduces KL from
0.1792153 to 0.1788410 (0.21%), improves 19/32 chunks, and has mean chunk reduction
0.0003743 with descriptive interval [-0.0017000, 0.0024486]. This does not establish a
portable gain. See `results/bf16_mean_control_q2/FROZEN_MEAN_CONTROL.md`.

## Completed: selective early-block precision

`build_hybrid_gguf.py` preserves the complete Q2 GGUF except for selected `blk.N.*`
tensors copied byte-for-byte from the shape-identical Q8 model. It validates identical
tensor names and shapes, reopens the result, and hashes every raw tensor payload against
its intended source.

On held-out prose positions, replacing blocks 0 and 1 recovers 7.19% of BF16-relative KL,
improves 29/32 chunks, adds 182.8 MiB (9.88% over Q2), and slows generation by 4.59%.
Replacing block 2 as well reaches 8.83%, but costs 330.7 MiB and slows generation 8.34%; its
F16-heavy donor tensors make the third block much less size-efficient. The two-block model
is therefore selected before examining its independent code logits. See
`results/bf16_hybrid_precision/HYBRID_PRECISION.md`.

Frozen on the existing Python code corpus, the preselected two-block model recovers 5.41%
of KL and 5.55% of JS, improves 27/32 chunks, and has a descriptive chunk-level KL interval
[0.001434, 0.017960]. The one- and three-block diagnostic points recover 1.64% and 6.79%,
respectively. Q4 recovers 92.87% but retains its much larger size and throughput costs.
This is the first cheap correction here with a clear gain on both workloads, although the
code interval still reuses chunks from one file rather than independent code projects. See
`results/bf16_hybrid_precision/FROZEN_HYBRID_PRECISION.md`.

## Completed: donor precision comparison

Replacing the same blocks 0 and 1 from Q4 instead of Q8 adds only 66.9 MiB, versus
182.8 MiB for Q8. On prose, early-Q4-2 recovers 6.81% of KL—94.8% of the Q8 hybrid's
absolute KL reduction—and its 32-chunk interval excludes zero. On code it recovers 4.09%,
or 75.5% of the Q8 reduction, but its interval narrowly crosses zero. Both donor choices
slow generation by about 3.8% in a paired benchmark, so Q4 saves memory rather than time.

This adds a useful Pareto point: prefer Q4 donor blocks for the strictest memory budget and
Q8 when the stronger code evidence is worth another 115.9 MiB. The close prose result also
motivates splitting blocks 0 and 1 by tensor family to find which upgrades carry the gain.
See `results/bf16_donor_precision/HYBRID_PRECISION.md`.

## Completed: early-block tensor families

The byte-verifying builder now accepts tensor-name regexes. Within Q4 donor blocks 0 and
1, the six `ffn_*` matrices add 38.7 MiB and recover 4.95% of prose KL, improving 28/32
chunks with interval [0.005767, 0.016164]. The recurrent family (`attn_gate`, `attn_qkv`,
and `ssm_*`) adds 28.3 MiB but recovers only 1.91%, improves 21/32 chunks, and has an
interval crossing zero. FFN delivers 1.90 times as much KL reduction per added MiB.

FFN-only was selected before loading family-level code logits. Frozen on code, it recovers
4.75% of KL, improves 28/32 chunks, and has interval [0.004977, 0.012055]. It slightly beats
the complete Q4 block's 4.09% code recovery despite using 28.3 MiB fewer, and retains 87.8%
of the Q8 block hybrid's reduction with 21.2% of its added bytes. Generation is 3.25% slower
than Q2 in a paired run; hybrid-to-hybrid differences are within run variability.

The FFN family is therefore validated well enough to split its up, gate, and down matrices.
See `results/bf16_tensor_families/FROZEN_HYBRID_PRECISION.md`.

## Completed provisionally: individual early FFN matrices

Across blocks 0 and 1, Q4 up-only recovers 2.96% of prose KL for 10.5 MiB, improves 24/32
chunks, and has interval [0.002439, 0.010691]. Gate-only recovers 2.34% for the same bytes,
improves 23/32 chunks, and also excludes zero. Down-only adds 17.6 MiB but worsens KL by
0.49%; its interval crosses zero. Up is the best individual memory point.

Because gate and up are both positive while down is negative, test the predeclared
21.1 MiB gate+up combination on prose. Only then select up-only or gate+up for code; do not
use individual matrix code captures in that choice. See
`results/bf16_ffn_matrices/HYBRID_PRECISION.md`.

## Completed: gate+up combination

The predeclared Q4 gate+up model replaces four matrices across blocks 0 and 1 and adds
21.1 MiB (1.14%) to Q2. It recovers 5.02% of prose KL, improves 27/32 chunks, and has
interval [0.005122, 0.017094]. It slightly beats the complete 38.7 MiB FFN upgrade's 4.95%,
confirming that the Q4 down matrices are unnecessary for this workload.

Gate+up is frozen as the primary sub-25-MiB design for code. Up-only remains a secondary
10.5 MiB memory endpoint; code must not be used to switch the primary designation.

On frozen code, gate+up recovers 3.03% of KL, improves 24/32 chunks, and has interval
[0.001743, 0.009104]. Up-only drops to 1.28% and crosses zero, so gate is needed for the
portable gain. Gate+up retains 55.9% of the Q8 block hybrid's code KL reduction with only
11.5% of its added bytes. Generation is 0.76% below Q2 in a paired run, within roughly one
run standard deviation. See `results/bf16_ffn_gate_up/FROZEN_HYBRID_PRECISION.md`.

## Completed: gate+up layer split

Q4 gate+up in block 1 alone adds 10.5 MiB (0.57%) and recovers 2.81% of prose KL, improving
24/32 chunks with interval [0.002977, 0.009454]. Block 0 alone recovers 2.31%, improves
23/32, and also excludes zero. The effects are approximately additive, but block 1 retains
56.0% of the two-block reduction for half its bytes and is the preselected single-block
design.

Frozen on code, block 1 recovers only 1.50%, improves 19/32 chunks, and has interval
[-0.000308, 0.005700]. It retains about half the two-block reduction but does not establish
a portable gain. A 20-repeat benchmark places block 1 and the two-block design 0.66% and
1.14% below Q2 generation speed, both smaller than run standard deviations. The two-block
21.1 MiB gate+up model remains the smallest supported configuration. See
`results/bf16_gate_up_layers/FROZEN_HYBRID_PRECISION.md`.

## Priority 2: larger-corpus adapter or LoRA distillation

The remaining learned-model test needs substantially more paired and more varied data.
Only then is it worth trying an earlier-layer state, residual head, last-block adapter,
or LoRA against BF16 soft targets.

Start with the least invasive version:

1. Expand paired prose and code calibration well beyond 2,016 positions.
2. Capture aligned Q2 final hidden states and BF16/Q2 logits.
3. Train a regularized map end-to-end against BF16 KL, optionally constraining its output
   to learned directions, rather than regressing squared error to oracle amplitudes.
4. Only if that transfers, move the map into a last-layer LoRA or adapter and benchmark
   Q2+adapter memory, tokens/s, and KL against Q3/Q4.

The local `llama-finetune` executable is not suitable as-is: it trains model weights on
hard next-token labels, rather than fitting a LoRA on a frozen quantized base against BF16
distributions. A real adapter experiment needs hidden-state export and a distillation
training path.

## Priority 3: token-feature correction

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
- Compare every target-aware oracle to equal-rank random and covariance-destroying controls;
  report the learned-specific gain separately from generic oracle capacity.

## Recommended next decisive experiment

Retain the two-block Q4 gate+up model as the compact recommendation. Before more
same-corpus tensor selection, validate it across multiple independently sourced prose and
code workloads; the current chunk intervals measure positions within only two files.
Separately, collect many independently sourced documents before adding learned model
capacity; more positions from the same transcript will not resolve workload-level
uncertainty. For another learned
test, optimize held-out BF16 KL directly rather than squared error to non-identifiable
per-row oracle coefficients, and retain equal-rank random controls. Reuse the hidden-state
extractor only after this data expansion. If a direct-KL final-state map is still rejected,
localize BF16/Q2 differences across earlier layers before attempting a last-block adapter
or LoRA. The existing output distributions and final states do not support a deployable
correction beyond the simple baselines.
