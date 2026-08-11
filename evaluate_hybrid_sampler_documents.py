#!/usr/bin/env python3
"""Evaluate one frozen hybrid temperature across whole documents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_sampler_grid import prepare_reference, sampler_metrics
from analyze_sparse import load_sparse
from evaluate_document_confirmation import mean_t_interval


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(documents: list[tuple[str, Path, Path, Path]], candidate_temperature: float,
             reference_temperature: float = 0.8,
             top_p: float = 0.95) -> tuple[list[dict], list[dict], dict]:
    if len(documents) < 2:
        raise ValueError("frozen evaluation requires at least two documents")
    labels = [label for label, _, _, _ in documents]
    if len(set(labels)) != len(labels):
        raise ValueError("document labels must be unique")
    rows = []
    chunk_rows = []
    for label, reference_path, base_path, candidate_path in documents:
        paired, chunks, _, n_chunks = load_sparse(reference_path, candidate_path)
        base_paired, base_chunks, _, base_n_chunks = load_sparse(reference_path, base_path)
        if base_n_chunks != n_chunks or not np.array_equal(base_chunks, chunks):
            raise ValueError(f"{label}: Q2 and hybrid captures are not chunk-aligned")
        reference = prepare_reference(paired, reference_temperature, top_p)
        base_reference = prepare_reference(base_paired, reference_temperature, top_p)
        orders = [np.argsort(-row.candidate) for row in paired]
        base_orders = [np.argsort(-row.candidate) for row in base_paired]
        selected = np.ones(len(paired), dtype=bool)
        q2 = sampler_metrics(
            base_paired, base_reference, base_orders, selected, reference_temperature, top_p)
        baseline = sampler_metrics(
            paired, reference, orders, selected, reference_temperature, top_p)
        corrected = sampler_metrics(
            paired, reference, orders, selected, candidate_temperature, top_p)
        reductions = []
        for chunk in range(n_chunks):
            chunk_selected = chunks == chunk
            raw_chunk = sampler_metrics(
                paired, reference, orders, chunk_selected, reference_temperature, top_p)
            corrected_chunk = sampler_metrics(
                paired, reference, orders, chunk_selected, candidate_temperature, top_p)
            reduction = raw_chunk["sampler_js"] - corrected_chunk["sampler_js"]
            reductions.append(reduction)
            chunk_rows.append({
                "document": label,
                "chunk": chunk,
                "positions": int(chunk_selected.sum()),
                "baseline_js": raw_chunk["sampler_js"],
                "corrected_js": corrected_chunk["sampler_js"],
                "js_reduction": reduction,
                "baseline_tv": raw_chunk["sampler_tv"],
                "corrected_tv": corrected_chunk["sampler_tv"],
            })
        reductions = np.asarray(reductions, dtype=np.float64)
        mean, low, high = mean_t_interval(reductions)
        rows.append({
            "document": label,
            "positions": len(paired),
            "chunks": n_chunks,
            "q2_js": q2["sampler_js"],
            "baseline_js": baseline["sampler_js"],
            "corrected_js": corrected["sampler_js"],
            "model_js_recovery": 1.0 - baseline["sampler_js"] / q2["sampler_js"],
            "js_reduction": baseline["sampler_js"] - corrected["sampler_js"],
            "js_recovery": 1.0 - corrected["sampler_js"] / baseline["sampler_js"],
            "total_js_recovery": 1.0 - corrected["sampler_js"] / q2["sampler_js"],
            "baseline_tv": baseline["sampler_tv"],
            "corrected_tv": corrected["sampler_tv"],
            "baseline_support": baseline["sampler_support_jaccard"],
            "corrected_support": corrected["sampler_support_jaccard"],
            "improved_chunks": int((reductions > 0).sum()),
            "chunk_interval_low": low,
            "chunk_interval_high": high,
        })

    weights = np.asarray([row["positions"] for row in rows], dtype=np.float64)
    q2_js = float(np.average([row["q2_js"] for row in rows], weights=weights))
    baseline_js = float(np.average([row["baseline_js"] for row in rows], weights=weights))
    corrected_js = float(np.average([row["corrected_js"] for row in rows], weights=weights))
    document_reductions = np.asarray([row["js_reduction"] for row in rows])
    mean, low, high = mean_t_interval(document_reductions)
    aggregate = {
        "documents": len(rows),
        "positions": int(weights.sum()),
        "improved_documents": int((document_reductions > 0).sum()),
        "pooled_q2_js": q2_js,
        "pooled_baseline_js": baseline_js,
        "pooled_corrected_js": corrected_js,
        "pooled_model_js_recovery": 1.0 - baseline_js / q2_js,
        "pooled_js_recovery": 1.0 - corrected_js / baseline_js,
        "pooled_total_js_recovery": 1.0 - corrected_js / q2_js,
        "mean_document_js_reduction": mean,
        "document_interval_low": low,
        "document_interval_high": high,
    }
    return rows, chunk_rows, aggregate


def write_report(path: Path, rows: list[dict], aggregate: dict,
                 candidate_temperature: float, top_p: float) -> None:
    lines = [
        "# Frozen hybrid temperature across documents",
        "",
        f"The compact hybrid and T={candidate_temperature:.4f}/top-p={top_p:.2f} were fixed",
        "before this evaluation. Every available out-of-selection document is retained.",
        "",
        "| Document | Q2 JS | Hybrid JS | +temperature JS | Model recovery | Temperature recovery | Total recovery | Better chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['document']} | {row['q2_js']:.7f} | {row['baseline_js']:.7f} | "
            f"{row['corrected_js']:.7f} | {row['model_js_recovery']:.2%} | "
            f"{row['js_recovery']:.2%} | {row['total_js_recovery']:.2%} | "
            f"{row['improved_chunks']}/{row['chunks']} |")
    lines += [
        "",
        f"Temperature compensation improves {aggregate['improved_documents']}/",
        f"{aggregate['documents']} whole documents. Across {aggregate['positions']} positions,",
        f"pooled hybrid JS falls from {aggregate['pooled_baseline_js']:.7f} to",
        f"{aggregate['pooled_corrected_js']:.7f}, recovering",
        f"{aggregate['pooled_js_recovery']:.2%}.",
        f"Raw Q2 pooled JS is {aggregate['pooled_q2_js']:.7f}; the model alone recovers",
        f"{aggregate['pooled_model_js_recovery']:.2%}, and model plus temperature recovers",
        f"{aggregate['pooled_total_js_recovery']:.2%} in total.",
        "",
        f"Mean whole-document JS reduction is {aggregate['mean_document_js_reduction']:.7f}",
        f"with descriptive document interval [{aggregate['document_interval_low']:.7f},",
        f"{aggregate['document_interval_high']:.7f}]. Document variation is the primary",
        "uncertainty summary; chunks are correlated positions within files.",
        "",
        "Aggregate results are in `frozen_hybrid_sampler_documents.csv`; chunk results are in",
        "`frozen_hybrid_sampler_document_chunks.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", nargs=4, action="append", default=[],
                        metavar=("LABEL", "REFERENCE", "Q2", "CANDIDATE"))
    parser.add_argument("--candidate-temperature", type=float, default=0.7625)
    parser.add_argument("--reference-temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_hybrid_sampler"))
    args = parser.parse_args()
    if len(args.document) < 2:
        parser.error("at least two --document entries are required")
    documents = [(label, Path(reference), Path(base), Path(candidate))
                 for label, reference, base, candidate in args.document]
    rows, chunks, aggregate = evaluate(
        documents, args.candidate_temperature, args.reference_temperature, args.top_p)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_hybrid_sampler_documents.csv", rows)
    write_csv(args.output_dir / "frozen_hybrid_sampler_document_chunks.csv", chunks)
    write_report(args.output_dir / "FROZEN_HYBRID_SAMPLER.md", rows, aggregate,
                 args.candidate_temperature, args.top_p)
    print(f"wrote {args.output_dir / 'FROZEN_HYBRID_SAMPLER.md'}")


if __name__ == "__main__":
    main()
