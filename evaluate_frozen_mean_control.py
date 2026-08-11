#!/usr/bin/env python3
"""Evaluate an all-training mean layer control vector on an OOD capture."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_sparse import load_sparse, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--state-layer", type=int, default=3)
    parser.add_argument("--control-layer", type=int, default=2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_mean_control_q2"))
    args = parser.parse_args()

    baseline_rows, baseline_chunks, _, n_chunks = load_sparse(args.reference, args.baseline)
    candidate_rows, candidate_chunks, _, candidate_n_chunks = load_sparse(
        args.reference, args.candidate)
    if candidate_n_chunks != n_chunks or not np.array_equal(candidate_chunks, baseline_chunks):
        raise ValueError("baseline and controlled captures are not chunk-aligned")
    selected = np.ones(len(baseline_rows), dtype=bool)
    raw = metrics(baseline_rows, selected, temperature=1.0)
    corrected = metrics(candidate_rows, selected, temperature=1.0)
    chunk_rows = []
    for chunk in range(n_chunks):
        chunk_selected = baseline_chunks == chunk
        for method, rows in (("identity", baseline_rows),
                             ("mean-layer-control", candidate_rows)):
            chunk_rows.append({"chunk": chunk, "method": method,
                               **metrics(rows, chunk_selected, temperature=1.0)})
    raw_chunks = {row["chunk"]: row["kl"] for row in chunk_rows
                  if row["method"] == "identity"}
    corrected_chunks = {row["chunk"]: row["kl"] for row in chunk_rows
                        if row["method"] == "mean-layer-control"}
    reductions = np.array([raw_chunks[chunk] - corrected_chunks[chunk]
                           for chunk in sorted(raw_chunks)], dtype=np.float64)
    reduction = float(reductions.mean())
    half_width = float(2.040 * reductions.std(ddof=1) / np.sqrt(len(reductions)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_mean_control_chunks.csv", chunk_rows)
    lines = [
        "# Frozen mean residual control on code",
        "",
        f"The all-prose mean BF16-minus-Q2 state vector at layer {args.state_layer} is frozen",
        f"and added after block {args.control_layer} at strength 1 before loading the Python-code",
        "capture. No code reference states or logits tune the 10 KiB sidecar.",
        "",
        "| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in (("Raw Q2", raw), ("Frozen mean layer control", corrected)):
        lines.append(
            f"| {label} | {result['kl']:.7f} | {1.0 - result['kl'] / raw['kl']:.2%} | "
            f"{result['js']:.7f} | {result['tv']:.7f} | "
            f"{result['top1_agreement']:.1%} | {result['top10_overlap']:.1%} |")
    lines += [
        "",
        f"The correction improves {int((reductions > 0).sum())}/{len(reductions)} chunks.",
        f"Mean per-chunk KL reduction is {reduction:.7f}; the descriptive chunk-level t interval is",
        f"[{reduction - half_width:.7f}, {reduction + half_width:.7f}].",
        "",
        "Per-chunk metrics are in `frozen_mean_control_chunks.csv`.",
        "",
    ]
    (args.output_dir / "FROZEN_MEAN_CONTROL.md").write_text("\n".join(lines))
    print(f"wrote {args.output_dir / 'FROZEN_MEAN_CONTROL.md'}")


if __name__ == "__main__":
    main()
