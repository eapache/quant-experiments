# Quantization sampler compensation experiments

This log records the real-model investigation so later sessions can continue without
reconstructing decisions from chat history.

## Goal

Test whether a small calibration set of paired next-token distributions can learn a
cheap post-logit transform that makes a heavily quantized model behave more like a
higher-precision reference before ordinary sampling is applied.

The nested hypotheses are:

1. A global logit scale (equivalently, a compensating temperature) explains a useful
   fraction of quantization drift.
2. Rank-shaped or token-specific corrections explain additional stable drift.
3. Corrections selected on a few contexts improve held-out contexts, rather than merely
   fitting calibration positions.
4. Any claimed correction remains useful at normal downstream temperatures and
   truncation settings.

## Prior artifacts

The `chats/` directory contains two earlier ChatGPT conversations and a synthetic proof
of concept under `chats/quant-sampler-compensation-lab/`. The synthetic study found that
temperature exactly repairs artificial logit scaling, while shrunken token bias and
predicted residual PCA directions repaired part of the error in a quantized micro-model.
It explicitly identified aligned real GGUFs as the missing decisive experiment.

## Available real-model environment

- llama.cpp checkout: `/home/eapache/src/llama.cpp`
- llama.cpp commit: `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`
- build revision reported by `llama-cli --version`: `10358 (030ebb558)`
- inference backend for this run: CPU (CUDA initialization reports no device)
- CPU: AMD Ryzen 9 9900X, 12 physical / 24 logical cores
- RAM: 60 GiB
- models, all from the same Qwen3.5-4B family:
  - `Qwen3.5-4B-BF16.gguf` (8,424,393,632 bytes; added for the latest round)
  - `Qwen3.5-4B-UD-Q8_K_XL.gguf` (5,952,048,288 bytes)
  - `Qwen3.5-4B-UD-Q4_K_XL.gguf` (2,912,109,728 bytes)
  - `Qwen3.5-4B-UD-Q2_K_XL.gguf` (1,940,825,248 bytes)

The first round had only Q8, Q4, and Q2, so it used Q8 as its experimental reference.
The latest round uses the subsequently added original BF16 GGUF as the reference.

Model SHA-256 hashes:

| File | SHA-256 |
|---|---|
| BF16 | `9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a` |
| Q8_K_XL | `e786a3c6570474c3885199bfb5adc54325aa7521a314e10b0aaefe16a54ba42f` |
| Q4_K_XL | `b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7` |
| Q2_K_XL | `79aa0b583c888976013002f66b9f617070d91cf8de6a09dc775983640ac59463` |

## Extraction decision

Use `llama-perplexity --kl-divergence-base FILE` separately on each quant with identical
input and context settings. Despite the option name, when `--kl-divergence` is absent it
writes a compact binary distribution file:

- 8-byte magic `_logits_`
- `uint32 n_ctx`, `int32 n_vocab`, `int32 n_chunk`
- `n_ctx * n_chunk` token IDs (`int32`)
- for each evaluated position in the latter half of each chunk:
  - two float32 values encoded in the first four uint16 slots: scale and minimum log-p
  - `n_vocab` uint16 values encoding clipped log-probabilities over a 16-log-unit range
  - one padding uint16 when needed for alignment

This is sufficient to reconstruct paired distributions from all three quants offline.
It avoids modifying llama.cpp and keeps inference and correction analysis separate.

Planned first pass: use Q8 as reference, compare Q4 and Q2 on identical 128-token chunks
from `chats/first.txt`, and split by whole chunk for calibration and validation. Whole-
chunk splits are important because adjacent token positions are correlated.

## Findings

### First paired extraction (2026-08-10)

Corpus: the first 1,024 tokens of `chats/first.txt`, divided into eight independent
128-token chunks. llama.cpp evaluates 63 positions from the latter half of each chunk,
giving 504 aligned prediction positions.

Commands (substitute `Q8`, `Q4`, or `Q2` in both model and output names):

```bash
/home/eapache/src/llama.cpp/build/bin/llama-perplexity \
  -m /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q8_K_XL.gguf \
  -f chats/first.txt -c 128 -b 128 -ub 128 --chunks 8 -t 12 \
  --no-warmup -ngl 0 --kl-divergence-base results/logits/q8.kld
```

Runtime was about 5 seconds for Q8, 12 seconds for Q4, and 15 seconds for Q2 when
Q4/Q2 ran concurrently on CPU. The compact distribution files are 239 MiB each and are
excluded from Git. They are reproducible with the command above.

The first decoder check found that exponentiating every stored entry produces about
1.023 total probability because entries below the 16-log-unit storage floor all share
the floor value. Treating code-zero entries as omitted instead retains:

| File | Mean retained mass | Minimum retained mass |
|---|---:|---:|
| Q8 | 0.999540 | 0.998023 |
| Q4 | 0.999532 | 0.997930 |
| Q2 | 0.999455 | 0.997462 |

The analyzer therefore uses the union of non-clipped reference and candidate entries and
renormalizes. This preserves at least 99.75% of the probability mass in every row tested.

### First cross-validation result

`analyze_real_logits.py` uses four outer folds. Each fold holds out two whole chunks (126
positions) and fits on the other six chunks (378 positions). No adjacent positions from a
held-out chunk appear in its calibration set.

At downstream T=1.0, mean held-out KL relative to Q8 was:

| Quant | Raw KL | Temperature KL | Temp + token bias KL | Fraction recovered |
|---|---:|---:|---:|---:|
| Q4_K_XL | 0.0171012 | 0.0171038 | 0.0166132 | 2.9% |
| Q2_K_XL | 0.2444413 | 0.2422055 | 0.2384249 | 2.5% |

The gap-bucket correction did not improve on temperature. Combining gap and token bias
recovered 2.8% for Q4 and 2.9% for Q2. Q4 top-1 agreement improved from 93.1% to 94.0%
with token bias; Q2 top-1 did not reliably improve.

For a reference T=0.8/top-p=0.95 nucleus sampler, held-out JS divergence was:

| Quant | Raw JS | Best tested JS | Method | Fraction recovered |
|---|---:|---:|---|---:|
| Q4_K_XL | 0.0074989 | 0.0073236 | temperature + token | 2.3% |
| Q2_K_XL | 0.0713439 | 0.0694271 | temperature + gap + token | 2.7% |

This proves that a small paired calibration can learn a correction that transfers to held-
out real-model contexts, but the useful fraction is small in this first corpus. The large
synthetic recoveries do not transfer to these GGUFs. Most Q2 error is token ranking/noise
that the tested low-capacity transforms cannot reconstruct.

### Temperature finding

When fitted to T=1 distributions, scale multipliers were extremely stable across folds:

| Quant | Mean scale | Fold SD | Equivalent quant T for reference T=0.8 |
|---|---:|---:|---:|
| Q4_K_XL | 1.001365 | 0.000781 | 0.798910 |
| Q2_K_XL | 1.033646 | 0.004385 | 0.773969 |

However, refitting the same scalar objective *at downstream T=0.8* gives:

| Quant | Mean scale | Fold SD |
|---|---:|---:|
| Q4_K_XL | 0.998228 | 0.000997 |
| Q2_K_XL | 0.994448 | 0.002225 |

The Q2 optimum changes from sharpening at T=1 to slight flattening at T=0.8. This is an
important negative result: real quantization drift is not a fixed global logit scale, so
there is no temperature correction that is simultaneously optimal for every downstream
temperature. Fitting at the actual sampler temperature is necessary when the goal is a
recommended sampling configuration. It also explains why the T=1-fitted scale slightly
worsened untruncated Q2 KL when evaluated at T=0.8 even though it improved top-p sampler
JS by 2%.

Full fold-level numbers are in `results/cross_validation.csv`, fitted scales in
`results/temperature_fits.csv`, and the generated summary in `results/REPORT.md`.

### Compute environment update

The sandboxed process cannot see NVIDIA devices, but an approved unsandboxed probe found:

- NVIDIA RTX PRO 4500 Blackwell, compute capability 12.0, 32,623 MiB
- NVIDIA GeForce RTX 3060, compute capability 8.6, 12,288 MiB

CPU inference is already fast for the 1,024-token extraction. GPU access will be useful
for expanded corpora or rollout experiments. Offline top-p analysis, not model inference,
was the slowest first-pass step because it sorts candidate support for every held-out row.
The analyzer now caches reference nucleus distributions and runs Q4/Q2 in two processes;
the full four-fold run takes about two minutes.

### Validation against llama.cpp

The custom parser was checked against llama.cpp's own `--kl-divergence` calculation on
the same CPU captures:

| Candidate | llama.cpp mean KL | Analyzer raw mean KL | Relative difference |
|---|---:|---:|---:|
| Q4_K_XL | 0.017134 | 0.017101 | -0.19% |
| Q2_K_XL | 0.244946 | 0.244441 | -0.21% |

The small difference is expected: llama.cpp ignores base terms below its storage cutoff,
while the analyzer takes the union of retained support and renormalizes. Agreement at the
0.2% level validates the binary decoder and metric implementation.

### Low-sample boundary on the first eight chunks

`analyze_low_sample.py` repeats the outer-fold evaluation with 63, 126, 252, and 378
calibration positions. The main findings are:

- Q2 temperature at T=1 recovers about 0.8% KL even with 63 positions. Its scale SD falls
  from 0.0203 at 63 positions to 0.0044 at 378.
- Q2 token bias overfits badly at 63-126 positions, becomes marginal around 252, and
  recovers 2.46% at 378.
- Q4 token bias is safer, improving KL by 0.5%, 1.1%, 2.2%, and 2.9% as calibration grows.
- At downstream T=0.8, scalar temperature is essentially useless for both quants.

This refutes the synthetic study's optimistic 8-16-position token-bias result for real
Q2 GGUF inference. A scalar is low-sample; vocabulary corrections are not.

### Backend identity

Running the same Q8 model and tokens on CUDA0 against the saved CPU Q8 distributions gave
mean KL 0.000561, RMS correct-token probability change 0.653%, and 99.008% top-1
agreement. This is only about 3.3% of Q4's Q8-relative KL, but it is measurable and causes
about 1% top-1 flips. The inference backend must be part of a calibration artifact's
identity. The expanded Q8/Q4/Q2 captures were therefore all regenerated on CUDA0.

### Expanded CUDA experiment

The RTX PRO 4500 generated 32 chunks / 2,016 evaluated positions per quant in roughly two
seconds per model. Each `.kld` file is 955 MiB. The corpus remains `chats/first.txt`, now
using its first 4,096 tokens.

Reference extraction command (repeat with Q4/Q2 model and output names):

```bash
/home/eapache/src/llama.cpp/build/bin/llama-perplexity \
  -m /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q8_K_XL.gguf \
  -f chats/first.txt -c 128 -b 128 -ub 128 --chunks 32 -t 12 --no-warmup \
  -dev CUDA0 -sm none -ngl all --kl-divergence-base results/logits/q8-32.kld
```

`analyze_sparse.py` stores only the union of non-clipped reference/candidate tokens per
position. On the original eight-chunk data it reproduced dense KL results within a few
millionths and reduced a representative run from minutes to 1.7 seconds. The full
32-chunk learning curve, including entropy-conditioned biases, takes about 45 seconds.

Four outer folds each hold out eight whole chunks (504 positions) and train on up to 24
other chunks (1,512 positions). At maximum calibration size:

| Quant / downstream T | Raw KL | Best token correction KL | Recovered |
|---|---:|---:|---:|
| Q4 / 0.8 | 0.0200073 | 0.0195165 | 2.45% |
| Q4 / 1.0 | 0.0171895 | 0.0167327 | 2.66% |
| Q2 / 0.8 | 0.2591990 | 0.2578914 | 0.50% |
| Q2 / 1.0 | 0.2222293 | 0.2186639 | 1.60% |

Q2 token bias does not become reliably beneficial until roughly 1,000 positions on these
broader held-out blocks. Entropy-conditioned token bias does not beat the single bias at
maximum calibration and is often worse. The error is not exposing a useful simple
entropy-dependent direction.

### Direct sampling-configuration result

`analyze_sampler_grid.py` targets the Q8 sampler at T=0.8/top-p=0.95. Each outer fold uses
only 256 evenly spaced positions from other chunks for configuration search and evaluates
on all 504 positions in the held-out block.

| Quant | Method | Candidate setting (fold mean) | Held-out JS | Recovered |
|---|---|---|---:|---:|
| Q4 | same settings | T=0.800, p=0.950 | 0.0075621 | 0% |
| Q4 | tuned T | T=0.795, p=0.950 | 0.0075797 | -0.23% |
| Q2 | same settings | T=0.800, p=0.950 | 0.0663932 | 0% |
| Q2 | tuned T | T=0.760, p=0.950 | 0.0649256 | 2.21% |
| Q2 | tuned p | T=0.800, p=0.931 | 0.0658770 | 0.78% |
| Q2 | jointly tuned | T=0.728, p=0.965 | 0.0646430 | 2.64% |

Q2 temperature-only fits are stable across folds: 0.755, 0.765, 0.755, and 0.760 (mean
0.760, sample SD about 0.004). Top-p-only fits are 0.930-0.935. Joint optima are less
identifiable: lower temperatures trade against larger nuclei, producing T=0.710-0.745
and top-p=0.960-0.970. The joint objective improves another 0.43 percentage points over
temperature-only, but its parameters are much less stable. Recommend the scalar T=0.76
result if a practical Q2 preset is desired.

Q4 correctly yields a null result: its optimum is already effectively the reference
sampler configuration, and selecting tiny apparent training improvements slightly hurts
held-out JS.

### Frozen out-of-domain validation

To separate real transfer from within-corpus selection, the settings above were frozen
and evaluated on the first 4,096 tokens of
`chats/quant-sampler-compensation-lab/quant_sampler_lab.py`. Fresh 32-chunk Q8/Q4/Q2
captures were generated on CUDA0 with the same extraction command, changing only `-f`
and the output names. No code-corpus position was used for fitting.

| Quant | Frozen method | Setting | Code JS | Recovered |
|---|---|---|---:|---:|
| Q2 | same settings | T=0.8, p=0.95 | 0.0514493 | 0% |
| Q2 | temperature | T=0.76, p=0.95 | 0.0510595 | 0.76% |
| Q2 | top-p | T=0.8, p=0.93125 | 0.0515347 | -0.17% |
| Q2 | joint | T=0.7275, p=0.965 | 0.0508212 | 1.22% |
| Q4 | temperature | T=0.795, p=0.95 | 0.0055471 | 0.38% |

Across 32 complete code chunks, the paired mean JS reduction from Q2 T=0.76 was
0.0003898 with a normal 95% interval [0.0000264, 0.0007532]. The joint setting reduced
JS by 0.0006281 [0.0002455, 0.0010107]. Q2 top-p alone reversed sign. Both frozen Q4
intervals include zero.

Thus the Q2 temperature direction transfers beyond the calibration domain, but its
relative recovery falls from 2.21% to 0.76%. Top-p is workload-specific. This strengthens
the scientific proof while weakening any claim of a universal preset: even for one exact
quant/backend, the magnitude depends on the context distribution.

### Overall conclusion

The technique is proven in the narrow scientific sense: a tiny paired calibration finds
a stable, quant-specific Q2 sampling adjustment that improves held-out distribution
matching, and correctly recommends no change for Q4. The effect is only 2-3%. Neither
ordinary samplers nor the tested token/gap/entropy corrections invert most of Q2's
quantization damage. The dominant residual is token-specific ranking noise not predictable
from the quantized output distribution by these low-capacity transforms.

### BF16 reference round

The original `Qwen3.5-4B-BF16.gguf` became available on 2026-08-10. Matching CUDA0
captures were generated for the 32-chunk prose and code corpora, changing the prior
extraction command's model and output to `bf16-32.kld` and `bf16-code32.kld`. Each capture
is 955 MiB. The BF16 file's SHA-256 is recorded above.

The established sparse and sampler analyses were generalized to label the reference and
include Q8 as a candidate:

```bash
python3 analyze_sparse.py \
  --reference results/logits/bf16-32.kld --reference-label BF16 \
  --q8 results/logits/q8-32.kld --q4 results/logits/q4-32.kld \
  --q2 results/logits/q2-32.kld --jobs 3 --output-dir results/bf16_gpu32

python3 analyze_sampler_grid.py \
  --reference results/logits/bf16-32.kld --reference-label BF16 \
  --q8 results/logits/q8-32.kld --q4 results/logits/q4-32.kld \
  --q2 results/logits/q2-32.kld --jobs 3 --output-dir results/bf16_gpu32
```

At T=1 and 1,512 calibration positions:

| Candidate | Raw KL to BF16 | Temp + token KL | Recovered | Top-1 agreement |
|---|---:|---:|---:|---:|
| Q8_K_XL | 0.0010029 | 0.0010081 | -0.51% | 98.1% |
| Q4_K_XL | 0.0165486 | 0.0160727 | 2.88% | 94.1% |
| Q2_K_XL | 0.2214713 | 0.2178065 | 1.65% | 77.8% |

The BF16 sampler target at T=0.8/top-p=0.95 produced:

| Candidate | Method | Mean candidate setting | Held-out JS | Recovered |
|---|---|---|---:|---:|
| Q8_K_XL | same settings | T=0.8000, p=0.9500 | 0.0008728 | 0% |
| Q4_K_XL | tuned T | T=0.7950, p=0.9500 | 0.0072793 | -0.14% |
| Q2_K_XL | tuned T | T=0.7625, p=0.9500 | 0.0648402 | 2.22% |
| Q2_K_XL | tuned jointly | T=0.7288, p=0.9650 | 0.0645550 | 2.65% |

This reproduces the earlier Q8-relative practical conclusion against the proper BF16
target. Q8 requires no sampler adjustment. The Q2 temperature shift is not an artifact
of using Q8 as teacher.

Settings were loaded directly from `results/bf16_gpu32/sampler_grid.csv`, frozen, and
evaluated on the independent code capture:

```bash
python3 evaluate_frozen_sampler.py \
  --reference results/logits/bf16-code32.kld --reference-label BF16 \
  --settings-csv results/bf16_gpu32/sampler_grid.csv \
  --q8 results/logits/q8-code32.kld --q4 results/logits/q4-code32.kld \
  --q2 results/logits/q2-code32.kld --output-dir results/bf16_ood_code
```

Q2 T=0.7625 reduced mean JS by 0.0004071 (0.79%), with a per-chunk normal 95%
interval [0.0000637, 0.0007504]. The frozen joint setting reduced JS by 0.0006456
(1.26%), interval [0.0002661, 0.0010251]. Q2 top-p alone again reversed sign. Q4's
intervals include zero, and Q8's settings remain identical to the target settings.

### Predictive low-rank Q2 denoiser

`analyze_low_rank.py` implements the priority experiment from `NEXT_STEPS.md`. It learns
SVD directions on the union of the reference and candidate top-128 tokens after fitting
temperature and static token bias. Four outer folds hold out eight complete chunks each.
Within each outer training set, a separate six-chunk validation split selects 0, 1, 2, 4,
8, or 16 directions and ridge alpha. The predictor sees only candidate entropy,
probability mass, margins, and projections onto learned directions. Oracle amplitudes
minimize actual KL on each held-out reference row with a damped Newton solve and are
reported only as an unattainable diagnostic.

```bash
python3 analyze_low_rank.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld --candidate-name Q2_K_XL \
  --reference-label BF16 --output-dir results/bf16_low_rank_q2
```

| Method | Held-out KL | Recovered | Top-1 agreement |
|---|---:|---:|---:|
| identity | 0.2214648 | 0% | 77.3% |
| target temperature | 0.2196202 | 0.83% | 77.3% |
| temperature + token bias | 0.2178015 | 1.65% | 77.8% |
| quant-only predicted low rank | 0.2182810 | 1.44% | 78.0% |
| 16-direction oracle | 0.0950399 | 57.09% | 94.4% |

Inner validation chose zero directions in folds 0, 1, and 3, and one direction with
alpha 1000 in fold 2. At this stage the large oracle gap was interpreted as proving useful
low-rank head structure and motivating amplitude inference. The later matched-null analysis
below overturns most of that interpretation: random directions recover nearly as much,
so this experiment establishes only that the predictor is not deployable and does not beat
static token bias.

### Conditional empirical posterior (2026-08-11)

`analyze_posterior.py` implements the recommended uncertainty-aware test without assuming
independent token errors. Quant-only entropy, mass, and top-gap features retrieve
calibration rows with similar candidate head shapes. Their correlated rank-wise residual
vectors are either averaged before softmax or treated as empirical posterior samples whose
softmax distributions are averaged. Head sizes 32/64/128 and neighbor counts
8/16/32/64/128 are selected on inner chunk splits, including a null option.

```bash
python3 analyze_posterior.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld --candidate-name Q2_K_XL \
  --reference-label BF16 --output-dir results/bf16_posterior_q2
```

Both methods selected the null static-bias baseline in all four folds. The best forced
non-null setting always used head 128 and 128 neighbors. Forced conditional means worsened
KL from 0.2214648 raw to 0.2402280 (-8.47% recovery). Posterior prediction was much safer
at 0.2226350 (-0.53%), showing that averaging over uncertain corrections is preferable to
applying their mean, but it still failed to improve BF16 matching. Results are in
`results/bf16_posterior_q2/POSTERIOR.md`.

### Tiny nonlinear low-rank predictor (2026-08-11)

`analyze_neural_low_rank.py` tests whether modest nonlinear capacity can infer the oracle
amplitudes from substantially richer quant-logit features. A one-hidden-layer tanh network
receives the top-64 candidate gaps and probabilities plus the entropy/mass/margin and
residual-basis projection features. It predicts all 16 oracle amplitudes. Hidden width
8/32, weight decay 0.001/0.01/0.1, and 50/100/200/400 training steps are nested-selected.
The resulting networks contain 1,624 or 6,448 parameters.

```bash
python3 analyze_neural_low_rank.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld --candidate-name Q2_K_XL \
  --reference-label BF16 --output-dir results/bf16_neural_low_rank_q2
```

All four folds selected the null static-bias baseline. Every fold preferred the strongest
tested weight decay; widths and training durations were unstable. Forced use worsened KL
to 0.2364235 (-6.75% recovery) and top-10 overlap to 74.7%, while the identically fitted
16-direction oracle remained at 0.0950399 (57.09% recovery). This is not a full transformer
test, and its squared-error oracle-amplitude target is not equivalent to direct KL training,
but it is evidence against spending more capacity on that exact pipeline at the current
sample size. Results are in
`results/bf16_neural_low_rank_q2/NEURAL_LOW_RANK.md`.

The local `llama-finetune` tool was inspected as a possible LoRA route. This revision's
example uses `llama_opt_param_filter_all` and hard next-token training data, then saves a
full model. It does not provide the needed frozen-quant LoRA distillation against paired
BF16 distributions. A meaningful adapter experiment will require hidden-state export and
a custom soft-target training path.

### Cumulative-mass calibration (2026-08-11)

`analyze_mass_calibration.py` measures BF16 probability retained by prefixes sorted under
Q2, averages the monotonic reference-mass-versus-quant-mass curve on calibration chunks,
and inverts it for a BF16 top-p=0.95 target. It also nested-selects 2/4/8 quant-entropy bins
against a global curve and imports the established scalar sampler settings for comparison.

```bash
python3 analyze_mass_calibration.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld --candidate-name Q2_K_XL \
  --reference-label BF16 \
  --sampler-settings-csv results/bf16_gpu32/sampler_grid.csv \
  --output-dir results/bf16_mass_q2
```

The global curve consistently implied candidate top-p 0.9364-0.9405 (mean 0.9387) and
recovered 0.64% held-out prose JS. Entropy conditioning was enabled in only one fold and
recovered 0.59% after null selection; forced conditioning recovered -0.03%. The global
mapping retained 94.99% untruncated BF16 mass in the Q2 support, close to the requested
95%, but remained well behind tuned temperature's 2.22% JS recovery.

The global model was then fitted on all 2,016 prose positions and frozen at top-p=0.938814
before reading the code capture:

```bash
python3 evaluate_frozen_mass.py \
  --train-reference results/logits/bf16-32.kld \
  --train-candidate results/logits/q2-32.kld \
  --ood-reference results/logits/bf16-code32.kld \
  --ood-candidate results/logits/q2-code32.kld \
  --candidate-name Q2_K_XL --reference-label BF16 \
  --sampler-settings-csv results/bf16_gpu32/sampler_grid.csv \
  --output-dir results/bf16_mass_q2
```

On code it recovered only 0.06% JS, with mean per-chunk reduction 0.0000284 and 95%
interval [-0.0002487, 0.0003055]. Temperature still recovered 0.79% with an interval above
zero. Cumulative-mass calibration therefore improves support calibration in-domain but is
not a portable correction on this evidence. See `results/bf16_mass_q2/MASS_CALIBRATION.md`
and `results/bf16_mass_q2/FROZEN_MASS.md`.

### Monotonic rank-conditioned gap calibration (2026-08-11)

`analyze_gap_calibration.py` tests the last planned sampler-only correction. Starting from
temperature plus static token bias, it fits weighted monotonic piecewise-linear mappings
from Q2 gaps to BF16 gaps. The direct method maps gaps from the candidate top token. The
pairwise method maps all top-head pairs, reconstructs one consistent logit vector, and
blends it with the baseline. Head sizes 8/16/32, 4/8/16/32 knots, 1/4 rank bins, and
strengths 0.05/0.1/0.25/0.5/1 are nested-selected with a null option.

```bash
python3 analyze_gap_calibration.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld --candidate-name Q2_K_XL \
  --reference-label BF16 --output-dir results/bf16_gap_q2
```

Direct calibration worsened held-out KL to 0.2180525 after selection, below the 1.65%
static-bias recovery. Selected pairwise calibration reached KL 0.2177002, or 1.70%
recovery, an incremental reduction of only 0.0001013. Its normal 95% interval across the
four outer blocks is [-0.0000102, 0.0002128]. Capacity is unstable: selected strengths
range from 0.05 to 0.5 and head sizes from 8 to 32.

The modal pairwise capacity (head 8, 32 knots, one rank bin) and median accepted strength
0.1 were fitted on all prose positions and frozen before loading code:

```bash
python3 evaluate_frozen_gap.py \
  --train-reference results/logits/bf16-32.kld \
  --train-candidate results/logits/q2-32.kld \
  --ood-reference results/logits/bf16-code32.kld \
  --ood-candidate results/logits/q2-code32.kld \
  --reference-label BF16 --output-dir results/bf16_gap_q2
```

On code, the gap layer reduces KL by 0.0000511 relative to its frozen static-bias
baseline, with a per-chunk interval [-0.0001899, 0.0002920] and 17/32 improved chunks.
The larger finding is that the prose-trained static token bias itself worsens raw code KL
from 0.1792113 to 0.1876659 (-4.72% recovery). Monotonic gap calibration is therefore not
a repeatable or portable improvement. This exhausts the planned methods that consume only
the existing saved logits; the next experiment should expose the final hidden state and
distill BF16 residual amplitudes from substantially more paired data.

### Final-hidden-state residual map (2026-08-11)

The installed llama.cpp revision exposes Qwen3.5's normalized final hidden state through
`llama_set_embeddings_nextn`. `extract_hidden_states.cpp` is a repository-local extractor
that reads the exact token stream and chunk layout from a `.kld` header, repeats the
perplexity tool's BOS and latter-half output rules, and writes 2,560 float32 features for
each of the 63 evaluated positions per chunk. Device selection is explicitly restricted
to CUDA0 with split mode `none`, matching the existing captures. The 2,016-row artifact is
about 20 MiB and is ignored as reproducible intermediate data.

```bash
./build_hidden_extractor.sh
./extract_hidden_states \
  --model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --tokens results/logits/q2-32.kld \
  --output results/logits/q2-hidden-32.bin \
  --gpu-layers 999 --device CUDA0 --threads 12

.venv/bin/python analyze_hidden_adapter.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld \
  --hidden results/logits/q2-hidden-32.bin \
  --reference-label BF16 --candidate-label Q2_K_XL \
  --output-dir results/bf16_hidden_q2
```

`analyze_hidden_adapter.py` maps all hidden dimensions to 4, 8, or 16 oracle residual
amplitudes using kernel-form ridge regression. Four outer folds hold out eight complete
chunks, while inner chunk splits select direction count and ridge alpha including a null
option. All four folds select null. The best forced non-null settings use alpha 100,000
and worsen KL in every fold: aggregate KL is 0.2313056 versus 0.2178012 for temperature
plus static token bias, or -4.44% recovery from raw. Top-10 overlap falls from 75.9% to
75.1%. The corresponding 16-direction oracle remains strong at KL 0.0952049 (57.01%
recovery). The later matched-null result shows that this does not by itself confirm
amplitude prediction as the unique bottleneck.

This is a first pass, not the originally proposed large-data adapter experiment: it still
uses only 2,016 paired prose positions. It is strong evidence against increasing model
capacity on this same calibration set. A follow-up should first expand paired prose well
beyond 2,016 positions, then reuse the extractor; if the final-state map remains rejected,
an earlier-layer state or true last-block/LoRA distillation is the more distinct test.

### Matched-null validation of the low-rank oracle (2026-08-11)

The original conclusion that 57.09% rank-16 oracle recovery proved useful low-rank
residual structure lacked an equal-capacity null. `analyze_low_rank_structure.py` now
compares the learned basis with three outer-fold-clean controls, always allowing BF16 to
optimize the same number of held-out per-row amplitudes:

- Gaussian random subspaces on the same training-token vocabulary;
- learned PCA directions with their vocabulary columns permuted, preserving their spectrum;
- PCA after independently randomizing every training residual sign, preserving token-wise
  magnitudes and sparsity while destroying signed covariance.

```bash
python3 analyze_low_rank_structure.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld \
  --output-dir results/bf16_low_rank_structure_q2 --jobs 2
```

The learned and Gaussian oracle rank curves are:

| Directions | Learned KL | Gaussian-null KL | Learned advantage |
|---:|---:|---:|---:|
| 1 | 0.1958070 | 0.1955216 | -0.0002855 |
| 2 | 0.1780480 | 0.1790250 | 0.0009770 |
| 4 | 0.1537440 | 0.1555497 | 0.0018058 |
| 8 | 0.1238309 | 0.1272242 | 0.0033933 |
| 16 | 0.0950399 | 0.0989945 | 0.0039546 |
| 32 | 0.0721075 | 0.0746878 | 0.0025804 |

At rank 16, learned PCA recovers 57.09% of raw KL, but Gaussian random directions already
recover 55.30%, sign-randomized PCA recovers 51.82%, and token-permuted PCA recovers
34.65%. Learned PCA beats every sampled rank-16 control in every outer fold, and its first
singular value is 1.66 times the sign-null mean, so the residual matrix does contain a
coherent token-aligned covariance signal. However, the learned advantage over the mean
Gaussian oracle is only 0.0039546 KL, 3.1% of the total learned-oracle reduction. Its
four-block t interval is [-0.0004270, 0.0083363].

The distribution explains why arbitrary directions are so powerful: BF16 has median
effective support 9.7 and mean top-16 mass 85.9%; Q2 has median effective support 15.1 and
mean top-16 mass 82.0%. Sixteen random directions plus target-aware coefficients can fit
much of this plausible-token head independently at every row.

The corrected conclusion is nuanced. There is suggestive transferable covariance, but no
sharp low-dimensional cutoff, and the learned rank curve closely follows the Gaussian
null through rank 32. The prior 57% figure mostly measures generic per-row oracle capacity;
it does not prove an intrinsically 16-dimensional residual or show that amplitude inference
is the sole missing ingredient. Future predictors should optimize BF16 KL directly and all
oracle claims should retain equal-rank random controls. Full results are in
`results/bf16_low_rank_structure_q2/LOW_RANK_STRUCTURE.md`.

The prose-trained bases were then frozen before loading the existing code captures:

```bash
python3 evaluate_frozen_low_rank_structure.py \
  --train-reference results/logits/bf16-32.kld \
  --train-candidate results/logits/q2-32.kld \
  --ood-reference results/logits/bf16-code32.kld \
  --ood-candidate results/logits/q2-code32.kld \
  --output-dir results/bf16_low_rank_structure_q2
```

The OOD result reverses the small in-domain learned advantage at every rank:

| Directions | Learned KL | Gaussian-null KL | Learned advantage |
|---:|---:|---:|---:|
| 1 | 0.1523397 | 0.1501769 | -0.0021628 |
| 2 | 0.1369415 | 0.1288493 | -0.0080922 |
| 4 | 0.1156910 | 0.1053922 | -0.0102988 |
| 8 | 0.0942480 | 0.0832724 | -0.0109756 |
| 16 | 0.0771405 | 0.0666812 | -0.0104593 |
| 32 | 0.0613929 | 0.0545914 | -0.0068015 |

At rank 16, learned PCA recovers 56.96% of raw code KL, while Gaussian directions recover
62.79%. Learned PCA wins only 1/32 chunks. Its mean advantage is -0.0104593 with a
chunk-level t interval [-0.0137749, -0.0071436]. Those chunks still come from one reused
code file, so this interval is descriptive, but the reversal is large and consistent.

Final conclusion: the prose residual matrix has covariance beyond a sign-randomized null,
but it is workload-specific and not demonstrated to be intrinsically low-dimensional.
Nearly all of the original oracle recovery comes from fitting arbitrary target-aware
directions to a concentrated head. Learned residual PCA is less portable than a random
subspace here. See
`results/bf16_low_rank_structure_q2/FROZEN_LOW_RANK_STRUCTURE.md`.

### Sparse high-precision output-head sidecar (2026-08-11)

The Q2_K_XL GGUF keeps its tied `token_embd.weight` output tensor at Q6_K. To isolate how
much error remains in that final matrix, `analyze_output_head.py` reads matching Q6_K and
BF16 tensor rows directly from the GGUFs and applies `(W_BF16 - W_Q6) h` to only the Q2
candidate's top-N logits. It uses the previously captured normalized final hidden states.
Head sizes 8 through 256 and nonnegative blend strengths are selected on an inner chunk
split for each outer fold.

```bash
PYTHONPATH=/home/eapache/src/llama.cpp/gguf-py python3 analyze_output_head.py \
  --reference results/logits/bf16-32.kld \
  --candidate results/logits/q2-32.kld \
  --hidden results/logits/q2-hidden-32.bin \
  --reference-model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-BF16.gguf \
  --candidate-model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --output-dir results/bf16_output_head_q2
```

Nested selection reduces held-out KL from 0.2214713 to 0.2202139, recovering 0.57%.
All four folds improve; their mean reduction is 0.0012574 with a four-block t interval
[0.0003679, 0.0021468]. Top-1 agreement rises from 77.3% to 77.8%, but top-10 overlap is
unchanged. Fixed strength 1 with head 256, which requires no BF16-logit calibration,
recovers 0.51% and also improves every block.

The alignment diagnostic recomputes the saved Q2 logits from Q2 final states and Q6_K
weights to 0.027 centered RMSE over the selected top-256 vocabulary rows. The actual head
correction has RMS 0.036, versus 0.959 for the complete BF16-Q2 residual on those tokens,
and their correlation is only 0.037. The transformer therefore creates nearly all of the
Q2 error before the output head. A full BF16 head sidecar would add about 1.18 GiB to the
1.81 GiB Q2 model; loading only the 26,172 rows touched by this corpus is not a portable
deployment strategy. The output-head upgrade is measurable but too small and costly to
recommend over sampler temperature compensation. Full results are in
`results/bf16_output_head_q2/OUTPUT_HEAD.md`.

The natural, calibration-free strength 1 and prose-tested head size 256 were frozen before
loading the existing code capture. An aligned code hidden-state artifact was generated with:

```bash
./extract_hidden_states \
  --model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --tokens results/logits/q2-code32.kld \
  --output results/logits/q2-code-hidden-32.bin \
  --gpu-layers 999 --device CUDA0 --threads 12

PYTHONPATH=/home/eapache/src/llama.cpp/gguf-py python3 evaluate_frozen_output_head.py \
  --reference results/logits/bf16-code32.kld \
  --candidate results/logits/q2-code32.kld \
  --hidden results/logits/q2-code-hidden-32.bin \
  --reference-model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-BF16.gguf \
  --candidate-model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --head 256 --strength 1 --output-dir results/bf16_output_head_q2
```

Code KL falls from 0.1792153 to 0.1789424, only 0.15% recovery. The correction improves
20/32 chunks; mean per-chunk reduction is 0.0002729 with a descriptive t interval
[-0.0001954, 0.0007413]. Top-1 rises by 0.1 point and top-10 is unchanged. The head-target
correlation also falls from 0.037 on prose to 0.028 on code. This independent workload does
not establish a portable benefit. See
`results/bf16_output_head_q2/FROZEN_OUTPUT_HEAD.md`.

### Residual-stream drift localization (2026-08-11)

The output-head result places nearly all Q2 degradation upstream, so the next experiment
captures the input residual stream of every transformer block. `extract_layer_states.cpp`
uses the repository llama.cpp revision's asynchronous layer-input hook and preserves the
same token, chunk, BOS, and evaluated-position rules as the logits and final-state captures.
The two 630 MiB float32 intermediates are ignored and reproducible:

```bash
./build_layer_extractor.sh

./extract_layer_states \
  --model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --tokens results/logits/q2-32.kld --output results/logits/q2-layers-32.bin \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --gpu-layers 999 --device CUDA0 --threads 12

# Repeat with the BF16 model, logits token header, and bf16-layers-32.bin.
python3 analyze_layer_drift.py \
  --reference-states results/logits/bf16-layers-32.bin \
  --candidate-states results/logits/q2-layers-32.bin \
  --reference-logits results/logits/bf16-32.kld \
  --candidate-logits results/logits/q2-32.kld \
  --output-dir results/bf16_layer_drift_q2
```

Relative residual-stream error rises from 1.79% at the Q6_K input embedding to 11.33%
after recurrent block 0, 16.29% after block 1, and 22.14% before the first full-attention
block. It falls slightly across that full-attention block to 21.39%, so the largest early
damage is in the first three recurrent gated-delta blocks rather than attention. Relative
error peaks at 42.35% at layer 17 and ends at 39.91% before the final block; absolute error
continues growing to 0.422 RMS because the residual stream itself grows.

The best global candidate scale barely reduces relative error at every layer, excluding a
simple magnitude mismatch. A chunk-held-out mean state bias recovers 9.15% of state MSE at
layer 3 and generally 3-5% later. Per-row state error is uncorrelated with final KL early,
then rises to correlation 0.27 by layers 30-31. The next cheap intervention is therefore a
10 KiB mean residual control vector before the first full-attention block, evaluated with
outer-fold-clean prose vectors and then frozen on code. If it does not improve logits,
selectively higher precision for the first recurrent blocks is the better model-side test.
See `results/bf16_layer_drift_q2/LAYER_DRIFT.md`.

### Mean residual-stream control vector (2026-08-11)

The largest transferable mean-state component occurs at layer 3, immediately before the
first full-attention block. `build_mean_control_vectors.py` fits the mean BF16-minus-Q2
layer-3 input difference on the 24 outer-training chunks. It stores the 2,560 float32
values as `direction.2`, which llama.cpp adds after block 2. Each GGUF is 10,464 bytes;
strength is fixed at the natural value 1, and no BF16 output logits tune the vector.

```bash
PYTHONPATH=/home/eapache/src/llama.cpp/gguf-py python3 build_mean_control_vectors.py \
  --reference-states results/logits/bf16-layers-32.bin \
  --candidate-states results/logits/q2-layers-32.bin \
  --state-layer 3 --control-layer 2 --output-dir results/bf16_mean_control_q2

# Repeat for fold0 through fold3.
/home/eapache/src/llama.cpp/build/bin/llama-perplexity \
  -m /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  -f chats/first.txt -c 128 -b 128 -ub 128 --chunks 32 -t 12 --no-warmup \
  -dev CUDA0 -sm none -ngl all \
  --control-vector results/bf16_mean_control_q2/mean_control_layer2_fold0.gguf \
  --kl-divergence-base results/logits/q2-mean-control-fold0.kld

python3 evaluate_mean_control.py \
  --reference results/logits/bf16-32.kld --baseline results/logits/q2-32.kld \
  --candidates results/logits/q2-mean-control-fold0.kld \
               results/logits/q2-mean-control-fold1.kld \
               results/logits/q2-mean-control-fold2.kld \
               results/logits/q2-mean-control-fold3.kld \
  --state-layer 3 --control-layer 2 --output-dir results/bf16_mean_control_q2
```

The correction recovers 9.15% of held-out state MSE at layer 3 but only 0.02% of final KL:
0.2214713 falls to 0.2214191. It improves 2/4 folds, and mean reduction 0.0000522 has a
four-block interval [-0.0013435, 0.0014478]. TV improves from 0.2140513 to 0.2134115 and
top-10 overlap rises from 75.7% to 76.0%, but top-1 changes only 0.3 point. A stable mean
hidden error is therefore not aligned with the BF16 KL objective. Full results and the
deployable sidecars are in `results/bf16_mean_control_q2/`.

The exact source of the existing code captures, recovered and verified from their token
headers, is `chats/quant-sampler-compensation-lab/quant_sampler_lab.py`. The all-prose
vector was frozen and evaluated without code-side tuning:

```bash
/home/eapache/src/llama.cpp/build/bin/llama-perplexity \
  -m /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  -f chats/quant-sampler-compensation-lab/quant_sampler_lab.py \
  -c 128 -b 128 -ub 128 --chunks 32 -t 12 --no-warmup \
  -dev CUDA0 -sm none -ngl all \
  --control-vector results/bf16_mean_control_q2/mean_control_layer2_all.gguf \
  --kl-divergence-base results/logits/q2-code-mean-control.kld

python3 evaluate_frozen_mean_control.py \
  --reference results/logits/bf16-code32.kld \
  --baseline results/logits/q2-code32.kld \
  --candidate results/logits/q2-code-mean-control.kld \
  --state-layer 3 --control-layer 2 --output-dir results/bf16_mean_control_q2
```

Code KL falls from 0.1792153 to 0.1788410, or 0.21% recovery. Only 19/32 chunks improve;
mean per-chunk reduction is 0.0003743 with descriptive interval
[-0.0017000, 0.0024486]. The tiny state-side correction is not reliably portable. See
`results/bf16_mean_control_q2/FROZEN_MEAN_CONTROL.md`.

### Selective early-block precision (2026-08-11)

Q2 and Q8 contain the same 426 tensor names and shapes. `build_hybrid_gguf.py` copies Q2
metadata and all unselected tensor payloads, substitutes complete early `blk.N.*` payloads
from Q8, then reopens the result and SHA-256 verifies all 426 raw tensor regions against
the intended source. The three ignored, reproducible model artifacts were built with:

```bash
PYTHONPATH=/home/eapache/src/llama.cpp/gguf-py python3 build_hybrid_gguf.py \
  --base /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --donor /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q8_K_XL.gguf \
  --layers 0 1 --output results/models/Qwen3.5-4B-Q2-earlyQ8-2.gguf
```

The one-, two-, and three-block hybrids add 91.4, 182.8, and 330.7 MiB to the 1.808 GiB
Q2 file. The nonlinear third increment comes from F16 tensors in Q8 block 2. Matching
32-chunk prose captures were generated with the standard CUDA0 command and compared to the
existing BF16 and Q2 captures:

```bash
/home/eapache/src/llama.cpp/build/bin/llama-perplexity \
  -m results/models/Qwen3.5-4B-Q2-earlyQ8-2.gguf -f chats/first.txt \
  -c 128 -b 128 -ub 128 --chunks 32 -t 12 --no-warmup \
  -dev CUDA0 -sm none -ngl all \
  --kl-divergence-base results/logits/q2-earlyq8-2-32.kld

python3 analyze_hybrid_precision.py \
  --reference results/logits/bf16-32.kld \
  --base-logits results/logits/q2-32.kld \
  --base-model /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --variant early-Q8-1 results/logits/q2-earlyq8-1-32.kld results/models/Qwen3.5-4B-Q2-earlyQ8-1.gguf \
  --variant early-Q8-2 results/logits/q2-earlyq8-2-32.kld results/models/Qwen3.5-4B-Q2-earlyQ8-2.gguf \
  --variant early-Q8-3 results/logits/q2-earlyq8-3-32.kld results/models/Qwen3.5-4B-Q2-earlyQ8-3.gguf \
  --benchmark Q4_K_XL results/logits/q4-32.kld /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output-dir results/bf16_hybrid_precision
```

The hybrids recover 3.33%, 7.19%, and 8.83% of raw Q2 KL, improving 26, 29, and 30 of 32
chunks. All three descriptive chunk intervals exclude zero. Five-repeat `llama-bench`
measurements at pp128/tg32 show generation slowdowns of 2.54%, 4.59%, and 8.34%; prompt
throughput differences are within run variability. Q4 recovers 92.53% but adds 926.3 MiB
and slows generation 17.42%. The two-block hybrid is preselected for frozen code validation
because it offers the best measured KL reduction per total added byte.

Exact sizes and SHA-256 identities are recorded in
`results/bf16_hybrid_precision/hybrid_models.csv`; raw benchmark aggregates are in
`hybrid_throughput.csv`. The hybrid model files themselves remain ignored.

The two-block design was then kept fixed and all three hybrids were captured on the exact
source of the existing independent code baseline:

```bash
/home/eapache/src/llama.cpp/build/bin/llama-perplexity \
  -m results/models/Qwen3.5-4B-Q2-earlyQ8-2.gguf \
  -f chats/quant-sampler-compensation-lab/quant_sampler_lab.py \
  -c 128 -b 128 -ub 128 --chunks 32 -t 12 --no-warmup \
  -dev CUDA0 -sm none -ngl all \
  --kl-divergence-base results/logits/q2-code-earlyq8-2.kld

python3 evaluate_frozen_hybrid_precision.py \
  --reference results/logits/bf16-code32.kld \
  --baseline results/logits/q2-code32.kld \
  --variant early-Q8-1 results/logits/q2-code-earlyq8-1.kld \
  --variant early-Q8-2 results/logits/q2-code-earlyq8-2.kld \
  --variant early-Q8-3 results/logits/q2-code-earlyq8-3.kld \
  --benchmark Q4_K_XL results/logits/q4-code32.kld \
  --primary early-Q8-2 --output-dir results/bf16_hybrid_precision
```

The preselected early-Q8-2 hybrid reduces code KL from 0.1792153 to 0.1695180, recovering
5.41%; JS recovery is 5.55%. It improves 27/32 chunks, and mean chunk reduction 0.0096974
has a descriptive interval [0.0014342, 0.0179605]. Top-10 overlap rises from 75.6% to
76.3%, while top-1 is unchanged at 84.0%. The adjacent one- and three-block diagnostics
recover 1.64% and 6.79%; Q4 recovers 92.87%. The primary result transfers in direction and
meaningful magnitude, though one reused code file cannot establish project-level variance.

During evaluation, a code position with fewer than ten explicit KLD head tokens exposed an
old top-overlap edge case. `analyze_sparse.metrics` now compares the largest observable
top-k, up to ten, instead of failing or inventing an ordering among clipped tail ties.

### Q4 versus Q8 early-block donors (2026-08-11)

The block set was held fixed at 0 and 1 while donor precision changed. The generic hybrid
builder byte-verified a Q4-donor model using:

```bash
PYTHONPATH=/home/eapache/src/llama.cpp/gguf-py python3 build_hybrid_gguf.py \
  --base /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --donor /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --layers 0 1 --output results/models/Qwen3.5-4B-Q2-earlyQ4-2.gguf
```

Its SHA-256 is `f2a42dd7ef6e320a67ab51146b8693931fc8ff0da082d3ecf24f669a60e06621`.
The file is 2,011,010,368 bytes, only 70,185,120 bytes (66.9 MiB, 3.62%) over Q2.
Matching prose and code KLD captures use the same commands and sources as the preceding
hybrid experiment, substituting this model and `earlyq4` output names.

On prose, early-Q4-2 reduces KL from 0.2214713 to 0.2063805, recovering 6.81%; the Q8 donor
reaches 7.19%. Q4 therefore retains 94.8% of Q8's absolute KL reduction while using 36.6%
as many added bytes. It improves 26/32 chunks with interval [0.0097875, 0.0203941]. On code,
Q4 reduces KL from 0.1792153 to 0.1718936 (4.09% recovery), retaining 75.5% of Q8's code
reduction. It improves 24/32 chunks, but interval [-0.0006077, 0.0152511] crosses zero.

Five-repeat CUDA0 pp128/tg32 benchmarking measured Q2 at 5419.0/230.21 tok/s, early-Q4-2
at 5408.8/221.46, and early-Q8-2 at 5526.5/221.64. Prompt differences are within run
variation; Q4 and Q8 donor generation are effectively tied at -3.80% and -3.73% versus
Q2. Lower donor precision saves 115.9 MiB but not measurable runtime. Exact aggregate,
chunk, identity, and benchmark data are in `results/bf16_donor_precision/`.

### Early-block tensor-family ablation (2026-08-11)

`build_hybrid_gguf.py` now accepts repeatable `--include` regular expressions and records
them in GGUF metadata. Selection, output type, and every raw tensor payload remain verified.
Q4 donor blocks 0 and 1 were split into disjoint FFN and recurrent/SSM families:

```bash
PYTHONPATH=/home/eapache/src/llama.cpp/gguf-py python3 build_hybrid_gguf.py \
  --base /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  --donor /home/eapache/src/llama.cpp/models/custom/Qwen3.5-4B/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --layers 0 1 --include '\.ffn_' \
  --output results/models/Qwen3.5-4B-Q2-earlyQ4-2-ffn.gguf

# Repeat with --include '\.(attn_gate|attn_qkv|ssm_)' and an -ssm output name.
```

The FFN model replaces six tensors, adds 40,550,624 bytes (38.7 MiB), and has SHA-256
`8824df80008db8ae498b7a0443a2dc98a1abe76582d6413f297d1df16c818982`. The SSM model
replaces 18 tensors, adds 29,634,784 bytes (28.3 MiB), and has SHA-256
`f7c15d9273146974a5dce0a2dafd9814e4be9f662472d481b661d151301543d4`.

On the 32 prose chunks, FFN-only reduces KL from 0.2214713 to 0.2105060, recovering 4.95%.
It improves 28/32 chunks with interval [0.0057667, 0.0161639]. SSM-only reaches 0.2172442,
recovering 1.91%, and improves 21/32 chunks with interval [-0.0002241, 0.0086782]. FFN
carries 72.7% of the full Q4 hybrid's absolute KL reduction with 57.8% of its bytes, and
delivers 1.90 times the SSM family's KL reduction per added MiB. It is frozen for code
validation before splitting the three FFN matrix types.

The preselected FFN-only model was then captured on the unchanged code source:

```bash
/home/eapache/src/llama.cpp/build/bin/llama-perplexity \
  -m results/models/Qwen3.5-4B-Q2-earlyQ4-2-ffn.gguf \
  -f chats/quant-sampler-compensation-lab/quant_sampler_lab.py \
  -c 128 -b 128 -ub 128 --chunks 32 -t 12 --no-warmup \
  -dev CUDA0 -sm none -ngl all \
  --kl-divergence-base results/logits/q2-code-earlyq4-2-ffn.kld

python3 evaluate_frozen_hybrid_precision.py \
  --reference results/logits/bf16-code32.kld --baseline results/logits/q2-code32.kld \
  --variant Q4-FFN-only results/logits/q2-code-earlyq4-2-ffn.kld \
  --variant Q4-full-blocks results/logits/q2-code-earlyq4-2.kld \
  --benchmark Q8-full-blocks results/logits/q2-code-earlyq8-2.kld \
  --primary Q4-FFN-only --output-dir results/bf16_tensor_families
```

Code KL falls from 0.1792153 to 0.1706995, recovering 4.75%; JS recovery is 4.63%.
FFN-only improves 28/32 chunks with interval [0.0049765, 0.0120551]. It unexpectedly
outperforms the complete Q4 block upgrade on this file (4.09%, 24/32 chunks) and retains
87.8% of the Q8 block hybrid's absolute KL reduction while using 21.2% as many added bytes.

A paired five-repeat CUDA0 pp128/tg32 benchmark measured Q2 at 5429.5/230.00 tok/s,
FFN-only at 5467.6/222.54, complete Q4 blocks at 5407.9/226.47, and complete Q8 blocks at
5534.3/220.70. FFN-only generation is 3.25% below Q2. Differences among hybrids are smaller
than their run standard deviations, so there is no defensible family-level speed ordering.
Raw aggregates are in `results/bf16_tensor_families/family_throughput.csv`.
