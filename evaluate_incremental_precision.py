#!/usr/bin/env python3
"""Evaluate whether a nested precision extension improves an existing hybrid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_hybrid_precision import evaluate_capture
from evaluate_document_confirmation import mean_t_interval


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(reference: Path, base: Path, current: Path, extension: Path,
             current_label: str, extension_label: str) -> tuple[list[dict], list[dict], dict]:
    base_metrics, base_chunks, chunks = evaluate_capture(reference, base)
    current_metrics, current_chunks, _ = evaluate_capture(reference, current, chunks)
    extension_metrics, extension_chunks, _ = evaluate_capture(reference, extension, chunks)
    by_label = {
        "Q2 baseline": (base_metrics, base_chunks),
        current_label: (current_metrics, current_chunks),
        extension_label: (extension_metrics, extension_chunks),
    }
    summary = []
    for role, label in (("base", "Q2 baseline"), ("current", current_label),
                        ("extension", extension_label)):
        metrics, _ = by_label[label]
        summary.append({
            "label": label,
            "role": role,
            "kl": metrics["kl"],
            "kl_recovery_from_q2": 1.0 - metrics["kl"] / base_metrics["kl"],
            "js": metrics["js"],
            "js_recovery_from_q2": 1.0 - metrics["js"] / base_metrics["js"],
            "tv": metrics["tv"],
            "top1_agreement": metrics["top1_agreement"],
            "top10_overlap": metrics["top10_overlap"],
        })

    maps = {
        label: {row["chunk"]: row for row in per_chunk}
        for label, (_, per_chunk) in by_label.items()
    }
    chunk_rows = []
    incremental = []
    total = []
    for chunk in sorted(maps["Q2 baseline"]):
        base_kl = maps["Q2 baseline"][chunk]["kl"]
        current_kl = maps[current_label][chunk]["kl"]
        extension_kl = maps[extension_label][chunk]["kl"]
        extension_gain = current_kl - extension_kl
        incremental.append(extension_gain)
        total.append(base_kl - extension_kl)
        chunk_rows.append({
            "chunk": chunk,
            "baseline_kl": base_kl,
            "current_kl": current_kl,
            "extension_kl": extension_kl,
            "current_reduction_from_q2": base_kl - current_kl,
            "extension_reduction_from_q2": base_kl - extension_kl,
            "incremental_reduction": extension_gain,
        })
    incremental = np.asarray(incremental, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    inc_mean, inc_low, inc_high = mean_t_interval(incremental)
    total_mean, total_low, total_high = mean_t_interval(total)
    comparison = {
        "chunks": len(incremental),
        "incrementally_improved_chunks": int((incremental > 0).sum()),
        "incremental_mean_kl_reduction": inc_mean,
        "incremental_interval_low": inc_low,
        "incremental_interval_high": inc_high,
        "passes_incremental_gate": inc_low > 0,
        "total_improved_chunks": int((total > 0).sum()),
        "total_mean_kl_reduction": total_mean,
        "total_interval_low": total_low,
        "total_interval_high": total_high,
    }
    return summary, chunk_rows, comparison


def write_report(path: Path, summary: list[dict], comparison: dict,
                 current_label: str, extension_label: str,
                 added_mib: float) -> None:
    lines = [
        "# Incremental selective-precision test",
        "",
        f"The **{extension_label}** design was predeclared as a marginal {added_mib:.1f} MiB extension",
        f"to **{current_label}**. It passes only if its paired per-chunk KL reduction over the",
        "current model has a descriptive interval above zero.",
        "",
        "| Model | Role | KL | Recovery from Q2 | JS recovery | Top-1 | Top-10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['role']} | {row['kl']:.7f} | "
            f"{row['kl_recovery_from_q2']:.2%} | {row['js_recovery_from_q2']:.2%} | "
            f"{row['top1_agreement']:.1%} | {row['top10_overlap']:.1%} |")
    verdict = "passes" if comparison["passes_incremental_gate"] else "fails"
    lines += [
        "",
        f"The extension improves {comparison['incrementally_improved_chunks']}/{comparison['chunks']}",
        f"chunks versus the current model. Mean incremental KL reduction is",
        f"{comparison['incremental_mean_kl_reduction']:.7f}, with interval",
        f"[{comparison['incremental_interval_low']:.7f},",
        f"{comparison['incremental_interval_high']:.7f}]. It **{verdict}** the predeclared",
        "incremental gate.",
        "",
        "Aggregate values are in `incremental_precision.csv`; paired chunk values are in",
        "`incremental_precision_chunks.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--current-label", default="Q4 gate+up blocks 0-1")
    parser.add_argument("--extension-label", default="Q4 gate+up blocks 0-2")
    parser.add_argument("--added-mib", type=float, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_gate_up_block2"))
    args = parser.parse_args()
    summary, chunks, comparison = evaluate(
        args.reference, args.base, args.current, args.extension,
        args.current_label, args.extension_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "incremental_precision.csv", summary)
    write_csv(args.output_dir / "incremental_precision_chunks.csv", chunks)
    write_report(args.output_dir / "INCREMENTAL_PRECISION.md", summary, comparison,
                 args.current_label, args.extension_label, args.added_mib)
    print(f"wrote {args.output_dir / 'INCREMENTAL_PRECISION.md'}")


if __name__ == "__main__":
    main()
