#!/usr/bin/env python3
"""Measure BF16-relative quality versus size for selective-precision hybrids."""

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


def evaluate_capture(reference: Path, capture: Path,
                     expected_chunks: np.ndarray | None = None) -> tuple[dict, list[dict], np.ndarray]:
    rows, chunks, _, n_chunks = load_sparse(reference, capture)
    if expected_chunks is not None and not np.array_equal(chunks, expected_chunks):
        raise ValueError(f"{capture} is not chunk-aligned with the baseline")
    aggregate = metrics(rows, np.ones(len(rows), dtype=bool), temperature=1.0)
    chunk_metrics = []
    for chunk in range(n_chunks):
        chunk_metrics.append({"chunk": chunk,
                              **metrics(rows, chunks == chunk, temperature=1.0)})
    return aggregate, chunk_metrics, chunks


def analyze(reference: Path, base_logits: Path, base_model: Path,
            variants: list[tuple[str, Path, Path]],
            benchmarks: list[tuple[str, Path, Path]]) -> tuple[list[dict], list[dict]]:
    base, base_chunks, chunks = None, None, None
    base_metrics, base_chunk_metrics, chunks = evaluate_capture(reference, base_logits)
    base_size = base_model.stat().st_size
    base_by_chunk = {row["chunk"]: row for row in base_chunk_metrics}
    summary = [{
        "label": "Q2 baseline", "kind": "base", "model_bytes": base_size,
        "added_bytes": 0, "kl_recovery": 0.0, "js_recovery": 0.0,
        "mean_chunk_kl_reduction": 0.0, "chunk_interval_low": 0.0,
        "chunk_interval_high": 0.0, "improved_chunks": 0,
        **base_metrics,
    }]
    chunk_rows = [{"label": "Q2 baseline", **row} for row in base_chunk_metrics]
    previous_kl = base_metrics["kl"]
    previous_size = base_size
    for kind, entries in (("hybrid", variants), ("benchmark", benchmarks)):
        for label, logits_path, model_path in entries:
            aggregate, per_chunk, _ = evaluate_capture(reference, logits_path, chunks)
            size = model_path.stat().st_size
            reductions = np.array([
                base_by_chunk[row["chunk"]]["kl"] - row["kl"] for row in per_chunk
            ], dtype=np.float64)
            mean = float(reductions.mean())
            half_width = float(2.040 * reductions.std(ddof=1) / np.sqrt(len(reductions)))
            added_mib = (size - base_size) / 2**20
            record = {
                "label": label, "kind": kind, "model_bytes": size,
                "added_bytes": size - base_size,
                "kl_recovery": 1.0 - aggregate["kl"] / base_metrics["kl"],
                "js_recovery": 1.0 - aggregate["js"] / base_metrics["js"],
                "mean_chunk_kl_reduction": mean,
                "chunk_interval_low": mean - half_width,
                "chunk_interval_high": mean + half_width,
                "improved_chunks": int((reductions > 0).sum()),
                **aggregate,
            }
            if kind == "hybrid":
                record["marginal_kl_per_mib"] = ((previous_kl - aggregate["kl"])
                                                   / ((size - previous_size) / 2**20))
                previous_kl = aggregate["kl"]
                previous_size = size
            else:
                record["marginal_kl_per_mib"] = ""
            summary.append(record)
            chunk_rows.extend({"label": label, **row} for row in per_chunk)
            print(f"finished {label}: KL={aggregate['kl']:.7f}, added={added_mib:.1f} MiB",
                  flush=True)
    summary[0]["marginal_kl_per_mib"] = ""
    return summary, chunk_rows


def write_report(path: Path, summary: list[dict]) -> None:
    base = summary[0]
    lines = [
        "# Selective early-block precision",
        "",
        "The hybrids retain every tensor byte from Q2_K_XL except complete early recurrent",
        "blocks copied from a shape-identical higher-precision GGUF. The builder reopens each model",
        "and verifies every raw tensor payload against its intended source. This experiment was",
        "chosen from residual-state localization without inspecting hybrid output logits.",
        "",
        "| Model | Size | Added | KL | KL recovered | JS recovered | Top-1 | Top-10 | Better chunks | KL interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        size_gib = row["model_bytes"] / 2**30
        added_mib = row["added_bytes"] / 2**20
        interval = ("—" if row["kind"] == "base" else
                    f"[{row['chunk_interval_low']:.6f}, {row['chunk_interval_high']:.6f}]")
        better = "—" if row["kind"] == "base" else f"{row['improved_chunks']}/32"
        lines.append(
            f"| {row['label']} | {size_gib:.3f} GiB | {added_mib:+.1f} MiB | "
            f"{row['kl']:.7f} | {row['kl_recovery']:.2%} | {row['js_recovery']:.2%} | "
            f"{row['top1_agreement']:.1%} | {row['top10_overlap']:.1%} | {better} | {interval} |")
    hybrids = [row for row in summary if row["kind"] == "hybrid"]
    if hybrids:
        efficient = max(hybrids, key=lambda row: row["kl_recovery"] / max(row["added_bytes"], 1))
        best = min(hybrids, key=lambda row: row["kl"])
        lines += [
            "",
            f"The most size-efficient tested hybrid is **{efficient['label']}**, recovering",
            f"{efficient['kl_recovery']:.2%} KL for {efficient['added_bytes'] / 2**20:.1f} MiB.",
            f"The best absolute hybrid is **{best['label']}** at {best['kl_recovery']:.2%}",
            f"recovery. Raw Q2 KL is {base['kl']:.7f}.",
        ]
        if len(hybrids) >= 3:
            lines += [
                "",
                "Marginal KL reduction per added MiB for blocks 1, 2, and 3 is "
                + ", ".join(f"{row['marginal_kl_per_mib']:.8f}" for row in hybrids[:3])
                + ", respectively.",
            ]
    lines += [
        "",
        "Aggregate values are in `hybrid_precision.csv`; per-chunk metrics are in",
        "`hybrid_precision_chunks.csv`. Intervals are descriptive t intervals over 32 chunks",
        "from one prose corpus.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--variant", nargs=3, action="append", default=[],
                        metavar=("LABEL", "LOGITS", "MODEL"))
    parser.add_argument("--benchmark", nargs=3, action="append", default=[],
                        metavar=("LABEL", "LOGITS", "MODEL"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_hybrid_precision"))
    args = parser.parse_args()
    if not args.variant:
        parser.error("at least one --variant is required")
    variants = [(label, Path(logits), Path(model)) for label, logits, model in args.variant]
    benchmarks = [(label, Path(logits), Path(model))
                  for label, logits, model in args.benchmark]
    summary, chunks = analyze(args.reference, args.base_logits, args.base_model,
                              variants, benchmarks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "hybrid_precision.csv", summary)
    write_csv(args.output_dir / "hybrid_precision_chunks.csv", chunks)
    write_report(args.output_dir / "HYBRID_PRECISION.md", summary)
    print(f"wrote {args.output_dir / 'HYBRID_PRECISION.md'}")


if __name__ == "__main__":
    main()
