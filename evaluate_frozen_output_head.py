#!/usr/bin/env python3
"""Evaluate a prose-decided sparse output-head correction on an OOD capture."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_hidden_adapter import load_hidden
from analyze_output_head import (compute_corrections, diagnostics, evaluate,
                                 load_weight_rows, top_indices)
from analyze_sparse import load_sparse


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, raw: dict[str, float], corrected: dict[str, float],
                 chunk_rows: list[dict], diagnostics_: dict[str, float], head: int,
                 strength: float, unique_tokens: int) -> None:
    raw_chunks = {row["chunk"]: row["kl"] for row in chunk_rows
                  if row["method"] == "identity"}
    corrected_chunks = {row["chunk"]: row["kl"] for row in chunk_rows
                        if row["method"] == "output-head-sidecar"}
    reductions = np.array([raw_chunks[chunk] - corrected_chunks[chunk]
                           for chunk in sorted(raw_chunks)], dtype=np.float64)
    mean = float(reductions.mean())
    # 31 df; this interval is descriptive because all chunks come from one code file.
    half_width = float(2.040 * reductions.std(ddof=1) / np.sqrt(len(reductions)))
    lines = [
        "# Frozen output-head sidecar on code",
        "",
        f"This test freezes head size {head} and strength {strength:g} before loading the existing",
        "Python-code capture. It applies the exact BF16-minus-Q6_K tied-head correction from",
        "the GGUF weights to the captured Q2 final hidden state. No code reference logits tune",
        "the transform.",
        "",
        f"The evaluation touches {unique_tokens:,} distinct top-{head} vocabulary rows.",
        "",
        "| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (("Raw Q2", raw), ("Frozen output-head sidecar", corrected)):
        lines.append(
            f"| {label} | {metrics['kl']:.7f} | {1.0 - metrics['kl'] / raw['kl']:.2%} | "
            f"{metrics['js']:.7f} | {metrics['tv']:.7f} | "
            f"{metrics['top1_agreement']:.1%} | {metrics['top10_overlap']:.1%} |")
    lines += [
        "",
        f"The correction improves {int((reductions > 0).sum())}/{len(reductions)} chunks. Mean",
        f"per-chunk KL reduction is {mean:.7f}; the descriptive chunk-level t interval is",
        f"[{mean - half_width:.7f}, {mean + half_width:.7f}].",
        "",
        "## Alignment diagnostics",
        "",
        f"- Recomputed Q6_K-logit centered RMSE: {diagnostics_['quant_recompute_centered_rmse']:.6f}",
        f"- Output-head correction RMS: {diagnostics_['head_correction_rms']:.6f}",
        f"- Full BF16-vs-Q2 residual RMS: {diagnostics_['target_residual_rms']:.6f}",
        f"- Correction/residual correlation: {diagnostics_['head_target_correlation']:.4f}",
        "",
        "Per-chunk metrics are in `frozen_output_head_chunks.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--gguf-python-path", type=Path,
                        default=Path("/home/eapache/src/llama.cpp/gguf-py"))
    parser.add_argument("--head", type=int, default=256)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/bf16_output_head_q2"))
    args = parser.parse_args()
    if args.head <= 0 or args.strength < 0:
        parser.error("head must be positive and strength must be nonnegative")

    rows, chunks, _, n_chunks = load_sparse(args.reference, args.candidate)
    hidden = load_hidden(args.hidden, args.candidate)
    per_row_top, unique_tokens = top_indices(rows, args.head)
    reference_weights, _ = load_weight_rows(
        args.reference_model, unique_tokens, args.gguf_python_path)
    quant_weights, _ = load_weight_rows(
        args.candidate_model, unique_tokens, args.gguf_python_path)
    corrections = compute_corrections(
        rows, hidden, per_row_top, unique_tokens, reference_weights - quant_weights)
    diagnostics_ = diagnostics(rows, hidden, per_row_top, unique_tokens,
                               quant_weights, corrections)
    all_rows = np.ones(len(rows), dtype=bool)
    raw = evaluate(rows, all_rows, per_row_top, corrections, 0, 0.0)
    corrected = evaluate(
        rows, all_rows, per_row_top, corrections, args.head, args.strength)
    chunk_rows = []
    for chunk in range(n_chunks):
        selected = chunks == chunk
        for method, head, strength in (("identity", 0, 0.0),
                                       ("output-head-sidecar", args.head, args.strength)):
            chunk_rows.append({"chunk": chunk, "method": method, "head": head,
                               "strength": strength,
                               **evaluate(rows, selected, per_row_top, corrections,
                                          head, strength)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_output_head_chunks.csv", chunk_rows)
    write_report(args.output_dir / "FROZEN_OUTPUT_HEAD.md", raw, corrected, chunk_rows,
                 diagnostics_, args.head, args.strength, len(unique_tokens))
    print(f"wrote {args.output_dir / 'FROZEN_OUTPUT_HEAD.md'}")


if __name__ == "__main__":
    main()
