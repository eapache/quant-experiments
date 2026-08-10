# Real Qwen3.5-4B quantization compensation results

## Experimental design

Q8_K_XL is the available reference; Q4_K_XL and Q2_K_XL are candidates. All files were evaluated on identical teacher-forced tokens. Four outer folds each hold out two complete 128-token chunks, so adjacent held-out positions never leak into calibration. Each fold trains on 378 positions and tests on 126.

Saved log-probabilities are clipped 16 log units below the row maximum. The analysis uses the union of non-clipped reference/candidate support and renormalizes it. Retained mass is recorded in `results.json`.

Corrections are fit upstream of the ordinary sampler in nested order: a global logit scale, nine quant-logit-gap buckets, and a strongly shrunken per-token residual bias.

## Cross-validated distribution results

### Q2_K_XL at downstream temperature 0.8

| Method | KL mean | fold SD | Recovered vs identity | Top-1 agreement | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| identity | 0.2771172 | 0.0238615 | 0.0% | 79.6% | 74.8% |
| temperature | 0.2793845 | 0.0239479 | -0.8% | 79.6% | 74.8% |
| temperature+gap | 0.2805916 | 0.0241369 | -1.3% | 79.6% | 74.8% |
| temperature+token | 0.2749625 | 0.0222087 | 0.8% | 78.6% | 74.9% |
| temperature+gap+token | 0.2742854 | 0.0227357 | 1.0% | 79.8% | 75.0% |

### Q2_K_XL at downstream temperature 1

| Method | KL mean | fold SD | Recovered vs identity | Top-1 agreement | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| identity | 0.2444413 | 0.0233779 | 0.0% | 79.6% | 74.8% |
| temperature | 0.2422055 | 0.0229005 | 0.9% | 79.6% | 74.8% |
| temperature+gap | 0.2423753 | 0.0229961 | 0.8% | 79.6% | 74.8% |
| temperature+token | 0.2384249 | 0.0226355 | 2.5% | 78.6% | 74.9% |
| temperature+gap+token | 0.2374077 | 0.0229022 | 2.9% | 79.8% | 75.0% |

### Q4_K_XL at downstream temperature 0.8

| Method | KL mean | fold SD | Recovered vs identity | Top-1 agreement | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| identity | 0.0194375 | 0.0011831 | 0.0% | 93.1% | 91.9% |
| temperature | 0.0194475 | 0.0011868 | -0.1% | 93.1% | 91.9% |
| temperature+gap | 0.0194666 | 0.0011064 | -0.1% | 93.1% | 91.9% |
| temperature+token | 0.0189266 | 0.0017344 | 2.6% | 94.0% | 91.9% |
| temperature+gap+token | 0.0189490 | 0.0016670 | 2.5% | 93.8% | 91.9% |

### Q4_K_XL at downstream temperature 1

| Method | KL mean | fold SD | Recovered vs identity | Top-1 agreement | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| identity | 0.0171012 | 0.0012457 | 0.0% | 93.1% | 91.9% |
| temperature | 0.0171038 | 0.0012590 | -0.0% | 93.1% | 91.9% |
| temperature+gap | 0.0171116 | 0.0012002 | -0.1% | 93.1% | 91.9% |
| temperature+token | 0.0166132 | 0.0016492 | 2.9% | 94.0% | 91.9% |
| temperature+gap+token | 0.0166211 | 0.0015923 | 2.8% | 93.8% | 91.9% |

## Nucleus sampler matching

Jensen-Shannon divergence after applying the same temperature and top-p cutoff to the Q8 reference and corrected quant distributions:

| Quant | Temperature | Method | sampler JS | Recovered vs identity | Support Jaccard |
|---|---:|---|---:|---:|---:|
| Q2_K_XL | 0.8 | identity | 0.0713439 | 0.0% | 59.9% |
| Q2_K_XL | 0.8 | temperature | 0.0699457 | 2.0% | 62.7% |
| Q2_K_XL | 0.8 | temperature+gap | 0.0698856 | 2.0% | 62.7% |
| Q2_K_XL | 0.8 | temperature+token | 0.0701333 | 1.7% | 61.2% |
| Q2_K_XL | 0.8 | temperature+gap+token | 0.0694271 | 2.7% | 61.7% |
| Q4_K_XL | 0.8 | identity | 0.0074989 | 0.0% | 87.4% |
| Q4_K_XL | 0.8 | temperature | 0.0074844 | 0.2% | 87.4% |
| Q4_K_XL | 0.8 | temperature+gap | 0.0075414 | -0.6% | 87.3% |
| Q4_K_XL | 0.8 | temperature+token | 0.0073236 | 2.3% | 87.6% |
| Q4_K_XL | 0.8 | temperature+gap+token | 0.0073461 | 2.0% | 87.5% |

## Fitted temperature stability

The fitted scale multiplies quant logits. To emulate a desired temperature of 0.8 using only an ordinary temperature setting, use `0.8 / scale`.

| Quant | Scale mean | fold SD | Equivalent T mean | fold SD |
|---|---:|---:|---:|---:|
| Q2_K_XL | 1.033646 | 0.004385 | 0.773969 | 0.003283 |
| Q4_K_XL | 1.001365 | 0.000781 | 0.798910 | 0.000623 |

## Interpretation

See `EXPERIMENT_LOG.md` for the running interpretation, limitations, and exact extraction commands. Raw fold-level metrics are in `cross_validation.csv` and fitted scales in `temperature_fits.csv`.
