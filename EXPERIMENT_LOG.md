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
- build revision reported by `llama-cli --version`: `10358 (030ebb558)`
- inference backend for this run: CPU (CUDA initialization reports no device)
- CPU: AMD Ryzen 9 9900X, 12 physical / 24 logical cores
- RAM: 60 GiB
- models, all from the same Qwen3.5-4B family:
  - `Qwen3.5-4B-UD-Q8_K_XL.gguf` (5,952,048,288 bytes)
  - `Qwen3.5-4B-UD-Q4_K_XL.gguf` (2,912,109,728 bytes)
  - `Qwen3.5-4B-UD-Q2_K_XL.gguf` (1,940,825,248 bytes)

Only Q8, Q4, and Q2 are available. Q8 is therefore the experimental reference, not a
claim of BF16 ground truth. Conclusions must be phrased as recovery toward Q8.

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
