#!/usr/bin/env python3
"""Evaluate a prose-selected selective-precision hybrid on a frozen corpus."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_hybrid_precision import evaluate_capture


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(reference: Path, baseline: Path,
             variants: list[tuple[str, Path]], primary: str) -> tuple[list[dict], list[dict]]:
    labels = [label for label, _ in variants]
    if len(set(labels)) != len(labels):
        raise ValueError("variant labels must be unique")
    if primary not in labels:
        raise ValueError(f"primary label {primary!r} is not one of {labels}")

    raw, raw_chunks, chunks = evaluate_capture(reference, baseline)
    raw_by_chunk = {row["chunk"]: row for row in raw_chunks}
    summary = [{
        "label": "Q2 baseline", "is_primary": False, "kl_recovery": 0.0,
        "js_recovery": 0.0, "mean_chunk_kl_reduction": 0.0,
        "chunk_interval_low": 0.0, "chunk_interval_high": 0.0,
        "improved_chunks": 0, **raw,
    }]
    chunk_rows = [{"label": "Q2 baseline", **row} for row in raw_chunks]

    for label, capture in variants:
        aggregate, per_chunk, _ = evaluate_capture(reference, capture, chunks)
        reductions = np.array([
            raw_by_chunk[row["chunk"]]["kl"] - row["kl"] for row in per_chunk
        ], dtype=np.float64)
        mean = float(reductions.mean())
        half_width = float(2.040 * reductions.std(ddof=1) / np.sqrt(len(reductions)))
        summary.append({
            "label": label, "is_primary": label == primary,
            "kl_recovery": 1.0 - aggregate["kl"] / raw["kl"],
            "js_recovery": 1.0 - aggregate["js"] / raw["js"],
            "mean_chunk_kl_reduction": mean,
            "chunk_interval_low": mean - half_width,
            "chunk_interval_high": mean + half_width,
            "improved_chunks": int((reductions > 0).sum()),
            **aggregate,
        })
        chunk_rows.extend({"label": label, **row} for row in per_chunk)
    return summary, chunk_rows


def write_report(path: Path, summary: list[dict], primary: str) -> None:
    base = summary[0]
    selected = next(row for row in summary if row["label"] == primary)
    interval_result = ("excludes zero" if selected["chunk_interval_low"] > 0
                       else "crosses zero")
    lines = [
        "# Frozen selective-precision hybrid on code",
        "",
        f"**{primary}** was selected on prose for size efficiency before any hybrid code",
        "logits were loaded. Other block counts are diagnostic curve points only.",
        "",
        "| Model | Role | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["label"] == "Q2 baseline":
            role, better, interval = "baseline", "—", "—"
        else:
            role = "preselected" if row["is_primary"] else "diagnostic"
            better = f"{row['improved_chunks']}/32"
            interval = f"[{row['chunk_interval_low']:.6f}, {row['chunk_interval_high']:.6f}]"
        lines.append(
            f"| {row['label']} | {role} | {row['kl']:.7f} | {row['kl_recovery']:.2%} | "
            f"{row['js_recovery']:.2%} | {row['top1_agreement']:.1%} | "
            f"{row['top10_overlap']:.1%} | {better} | {interval} |")
    lines += [
        "",
        f"The preselected model changes code KL from {base['kl']:.7f} to",
        f"{selected['kl']:.7f}, recovering {selected['kl_recovery']:.2%}. It improves",
        f"{selected['improved_chunks']}/32 chunks; its descriptive chunk-level interval",
        f"{interval_result}: [{selected['chunk_interval_low']:.6f},",
        f"{selected['chunk_interval_high']:.6f}].",
        "",
        "Aggregate results are in `frozen_hybrid_precision.csv`; per-chunk metrics are in",
        "`frozen_hybrid_precision_chunks.csv`. All intervals reuse chunks from one code file",
        "and are descriptive rather than evidence across independent code workloads.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", nargs=2, action="append", default=[],
                        metavar=("LABEL", "LOGITS"))
    parser.add_argument("--primary", required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_hybrid_precision"))
    args = parser.parse_args()
    if not args.variant:
        parser.error("at least one --variant is required")
    variants = [(label, Path(capture)) for label, capture in args.variant]
    summary, chunks = evaluate(args.reference, args.baseline, variants, args.primary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_hybrid_precision.csv", summary)
    write_csv(args.output_dir / "frozen_hybrid_precision_chunks.csv", chunks)
    write_report(args.output_dir / "FROZEN_HYBRID_PRECISION.md", summary, args.primary)
    print(f"wrote {args.output_dir / 'FROZEN_HYBRID_PRECISION.md'}")


if __name__ == "__main__":
    main()
