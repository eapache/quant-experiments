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
alpha 1000 in fold 2. The large oracle gap proves that the Q2 error has useful low-rank
head structure, overturning the earlier tentative description of the residual as simply
irreducible noise. The tested quant-only features cannot predict its context-specific
amplitudes, so the low-rank model is not deployable and does not beat static token bias.
The next experiment should focus on amplitude inference or an uncertainty-aware head
sampler, not on adding more residual directions.

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
test, but it is evidence against spending more capacity on the same saved-logit inputs at
the current sample size. Results are in
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
