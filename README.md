# Quant sampler compensation experiments

This repository tests whether paired next-token distributions can predict better sampling
settings, or learn a cheap pre-sampler correction, for a quantized language model.

## Bottom line

The idea works, but only weakly for the tested Qwen3.5-4B GGUFs.

Using Q8_K_XL as the available reference and four-fold held-out evaluation over 2,016
teacher-forced positions:

- Q2_K_XL benefits from changing a Q8 sampler at T=0.8/top-p=0.95 to approximately
  **T=0.76/top-p=0.95**. Temperature is stable across folds (0.755-0.765) and recovers
  2.21% of Jensen-Shannon divergence to the reference sampler.
- Tuning top-p alone chooses about 0.93 and recovers 0.78%.
- Joint T/top-p tuning recovers 2.64%, but the parameters trade off along a ridge and are
  less stable, so the temperature-only recommendation is better supported.
- With settings frozen and moved to a Python-code corpus, Q2 T=0.76 still improves JS by
  0.76% (normal per-chunk 95% interval excludes zero). The effect transfers but shrinks.
- Q4_K_XL wants effectively the original T=0.8/top-p=0.95 settings. Tuning slightly
  worsens held-out divergence.
- A shrunken token-specific logit bias recovers about 2.5-2.7% of Q4 KL, but only 0.5-1.6%
  of Q2 KL depending on downstream temperature. An entropy-conditioned variant does not
  improve held-out results.
- Most Q2 distortion is therefore irreducible token/rank noise from the perspective of
  the tested sampler-side transforms.

These are corrections **toward Q8**, not BF16. They are tied to the exact model files,
llama.cpp revision, backend, prompt distribution, and reference sampler.

## Durable results

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

NumPy is the only non-standard Python dependency for the sparse analyses. The dense
analyzer also uses SciPy.
