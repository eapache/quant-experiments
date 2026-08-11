# Frozen external mixed-format confirmation corpus

This second confirmation set was fixed before generating any model captures. It is outside
the quant-experiments repository and changes both authorship and format:

| Document | Format | Source | Bytes | Lines |
|---|---|---|---:|---:|
| stdlib-argparse | Python | Python 3.12 standard library | 101,750 | 2,677 |
| llama-sampling | C++ | llama.cpp sampling implementation | 34,772 | 919 |
| server-docs | Markdown | llama.cpp server documentation | 100,824 | 2,069 |
| gpl3 | legal prose | system GPLv3 license text | 35,149 | 674 |

Every source is evaluated independently for 32 chunks at context 128. Exact absolute paths
and SHA-256 identities are in `confirmation_corpora.csv`. No source may be removed or
substituted based on its result. The candidate remains the already-selected 21.1 MiB Q4
gate+up hybrid in blocks 0 and 1.

The files are locally installed inputs rather than copied repository assets. Their hashes
make this run auditable, but another machine may have different versions or paths.

The completed result is in `DOCUMENT_CONFIRMATION.md`. All four formats improve and pooled
KL recovery is 2.66%, but the descriptive interval across four whole-document absolute
reductions crosses zero because effect magnitude varies substantially by format.
