# Recommended compact Qwen3.5-4B Q2 recovery

## Configuration

Start from Qwen3.5-4B UD-Q2_K_XL and replace only these four tensors with their exact
UD-Q4_K_XL payloads:

- `blk.0.ffn_gate.weight`
- `blk.0.ffn_up.weight`
- `blk.1.ffn_gate.weight`
- `blk.1.ffn_up.weight`

For a BF16 sampler target of T=0.8/top-p=0.95, run the hybrid at
**T=0.7625/top-p=0.95**. No hybrid-specific temperature is justified.

```bash
./build_recommended_hybrid.sh \
  /path/to/Qwen3.5-4B-UD-Q2_K_XL.gguf \
  /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  results/models/Qwen3.5-4B-Q2-Q4-gate-up-blocks0-1.gguf
```

The builder checks identical tensor names and shapes, writes all unselected tensors from
Q2, reopens the result, and SHA-256 verifies every raw tensor payload against its intended
source.

## Exact tested identities

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Q2 base | 1,940,825,248 | `79aa0b583c888976013002f66b9f617070d91cf8de6a09dc775983640ac59463` |
| Q4 donor | 2,912,109,728 | `b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7` |
| Recommended hybrid | 1,962,943,872 | `913dc3d9e89b560e4bd806f2cb65e9e7f9502cd2da80969b6e83029947462c5c` |

The hybrid adds 22,118,624 bytes: 21.1 MiB or 1.14% over Q2.

## Evidence

| Evaluation | Recovery | Breadth |
|---|---:|---|
| BF16-relative KL, selection prose | 5.02% | 27/32 chunks improve |
| BF16-relative KL, frozen code | 3.03% | 24/32 chunks; interval above zero |
| BF16-relative KL, repository confirmation | 4.02% pooled | 4/4 documents improve; document interval above zero |
| BF16-relative KL, external mixed formats | 2.66% pooled | 4/4 documents improve; document interval crosses zero |
| Sampler JS from raw Q2, frozen documents | 3.79% pooled total | 9/9 documents improve after model + temperature |
| Temperature contribution on hybrid | 1.06% pooled incremental | 9/9 documents; document interval above zero |

A 20-repeat CUDA0 pp128/tg32 comparison measured 5,759.6/225.84 tok/s for Q2 and
5,894.6/223.27 for the hybrid. The generation difference is -1.14%, smaller than run
standard deviations; no throughput penalty is established at this resolution.

## Why this point

- Q4 gate+up in only one block costs 10.5 MiB but does not establish code transfer.
- Adding Q4 down matrices raises overhead to 38.7 MiB and improves code KL further, but is
  less byte-efficient.
- Upgrading complete Q4 or Q8 blocks costs 66.9 or 182.8 MiB.
- Extending gate+up into block 2 costs another 11.6 MiB and fails its paired incremental
  prose gate.
- Refitting temperature specifically for the hybrid does not beat the existing Q2 setting.

This is the smallest tested model-side configuration with clear gains on both selection
and frozen workloads. It is specific to the exact Qwen3.5-4B GGUFs, llama.cpp revision
`030ebb558`, CUDA backend, context/prompt distributions, and BF16 sampler target above.
Validation spans Python, C++, Markdown, and legal prose, but not other model families,
languages, long contexts, or free-running generation quality.
