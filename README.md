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
- A top-128 low-rank Q2 residual model reveals a large diagnostic gap: oracle amplitudes
  for 16 learned directions recover 57.09% of KL, but a quant-only ridge predictor is
  rejected in three of four folds and recovers only 1.44%, worse than static token bias.
  The missing ingredient is predicting context-specific correction amplitudes, not finding
  residual structure.

The artifacts are tied to the exact model files, llama.cpp revision, CUDA backend, prompt
distribution, and reference sampler.

## Durable results

- [`NEXT_STEPS.md`](NEXT_STEPS.md): completed low-rank diagnostic and next quant-aware samplers
- [`results/bf16_gpu32/REPORT.md`](results/bf16_gpu32/REPORT.md): BF16-relative correction curves
- [`results/bf16_gpu32/SAMPLER.md`](results/bf16_gpu32/SAMPLER.md): BF16-relative sampler search
- [`results/bf16_low_rank_q2/LOW_RANK.md`](results/bf16_low_rank_q2/LOW_RANK.md): nested low-rank diagnostic
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

NumPy is the only non-standard Python dependency for the sparse analyses. The dense
analyzer also uses SciPy.
