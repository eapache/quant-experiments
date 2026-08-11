# Frozen multi-document confirmation corpus

This confirmation set was fixed before generating or loading any new BF16, Q2, or hybrid
captures. It contains four separate Python implementations already versioned in the
repository but unused as logit-evaluation inputs during hybrid selection:

| Document | Path | Bytes | Lines | Chunks |
|---|---|---:|---:|---:|
| real-logits | `analyze_real_logits.py` | 22,120 | 473 | 32 |
| low-rank-controls | `analyze_low_rank_structure.py` | 19,585 | 377 | 32 |
| gap-calibration | `analyze_gap_calibration.py` | 19,049 | 387 | 32 |
| hidden-adapter | `analyze_hidden_adapter.py` | 17,137 | 323 | 32 |

Each document is evaluated independently at context 128. The primary candidate is already
fixed as the Q4 `ffn_gate` + `ffn_up` tensors in blocks 0 and 1, adding 21.1 MiB to Q2.
No document may be removed or replaced based on its measured outcome. Exact source hashes
are in `confirmation_corpora.csv`.

This set is stronger than treating chunks from one file as independent workloads, but it
still represents one repository, one authoring style, and Python only. Its document-level
variation should therefore be reported directly rather than generalized to all workloads.

The completed frozen evaluation is in `DOCUMENT_CONFIRMATION.md`. All four documents
improve, pooled KL recovery is 4.02%, and the descriptive interval across whole-document
KL reductions excludes zero.
