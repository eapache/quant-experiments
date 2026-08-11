#!/usr/bin/env python3
"""Evaluate one frozen model choice across multiple whole documents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_hybrid_precision import evaluate_capture


def mean_t_interval(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean and a descriptive two-sided 95% t interval."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("an interval requires at least two values")
    # Two-sided 95% Student t critical values indexed by degrees of freedom.
    critical = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        31: 2.040,
    }
    degrees = len(values) - 1
    multiplier = critical.get(degrees, 1.960)
    mean = float(values.mean())
    half_width = float(multiplier * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half_width, mean + half_width


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(documents: list[tuple[str, Path, Path, Path]]) -> tuple[list[dict], list[dict], dict]:
    if len(documents) < 2:
        raise ValueError("confirmation requires at least two documents")
    labels = [label for label, _, _, _ in documents]
    if len(set(labels)) != len(labels):
        raise ValueError("document labels must be unique")

    rows = []
    chunk_rows = []
    for label, reference, baseline, candidate in documents:
        raw, raw_chunks, chunks = evaluate_capture(reference, baseline)
        corrected, corrected_chunks, _ = evaluate_capture(reference, candidate, chunks)
        raw_by_chunk = {row["chunk"]: row for row in raw_chunks}
        corrected_by_chunk = {row["chunk"]: row for row in corrected_chunks}
        reductions = np.array([
            raw_by_chunk[chunk]["kl"] - corrected_by_chunk[chunk]["kl"]
            for chunk in sorted(raw_by_chunk)
        ], dtype=np.float64)
        mean, low, high = mean_t_interval(reductions)
        record = {
            "document": label,
            "positions": len(chunks),
            "chunks": len(raw_chunks),
            "baseline_kl": raw["kl"],
            "candidate_kl": corrected["kl"],
            "kl_reduction": raw["kl"] - corrected["kl"],
            "kl_recovery": 1.0 - corrected["kl"] / raw["kl"],
            "baseline_js": raw["js"],
            "candidate_js": corrected["js"],
            "js_recovery": 1.0 - corrected["js"] / raw["js"],
            "baseline_tv": raw["tv"],
            "candidate_tv": corrected["tv"],
            "baseline_top1": raw["top1_agreement"],
            "candidate_top1": corrected["top1_agreement"],
            "baseline_top10": raw["top10_overlap"],
            "candidate_top10": corrected["top10_overlap"],
            "improved_chunks": int((reductions > 0).sum()),
            "chunk_interval_low": low,
            "chunk_interval_high": high,
        }
        rows.append(record)
        for chunk in sorted(raw_by_chunk):
            chunk_rows.append({
                "document": label,
                "chunk": chunk,
                "positions": int((chunks == chunk).sum()),
                "baseline_kl": raw_by_chunk[chunk]["kl"],
                "candidate_kl": corrected_by_chunk[chunk]["kl"],
                "kl_reduction": (raw_by_chunk[chunk]["kl"]
                                 - corrected_by_chunk[chunk]["kl"]),
                "baseline_js": raw_by_chunk[chunk]["js"],
                "candidate_js": corrected_by_chunk[chunk]["js"],
            })

    weights = np.array([row["positions"] for row in rows], dtype=np.float64)
    def pooled(key: str) -> float:
        return float(np.average([row[key] for row in rows], weights=weights))

    document_reductions = np.array([row["kl_reduction"] for row in rows])
    doc_mean, doc_low, doc_high = mean_t_interval(document_reductions)
    baseline_kl = pooled("baseline_kl")
    candidate_kl = pooled("candidate_kl")
    baseline_js = pooled("baseline_js")
    candidate_js = pooled("candidate_js")
    aggregate = {
        "documents": len(rows),
        "positions": int(weights.sum()),
        "improved_documents": int((document_reductions > 0).sum()),
        "pooled_baseline_kl": baseline_kl,
        "pooled_candidate_kl": candidate_kl,
        "pooled_kl_recovery": 1.0 - candidate_kl / baseline_kl,
        "pooled_baseline_js": baseline_js,
        "pooled_candidate_js": candidate_js,
        "pooled_js_recovery": 1.0 - candidate_js / baseline_js,
        "pooled_baseline_tv": pooled("baseline_tv"),
        "pooled_candidate_tv": pooled("candidate_tv"),
        "pooled_baseline_top1": pooled("baseline_top1"),
        "pooled_candidate_top1": pooled("candidate_top1"),
        "pooled_baseline_top10": pooled("baseline_top10"),
        "pooled_candidate_top10": pooled("candidate_top10"),
        "mean_document_kl_reduction": doc_mean,
        "document_interval_low": doc_low,
        "document_interval_high": doc_high,
    }
    return rows, chunk_rows, aggregate


def write_report(path: Path, rows: list[dict], aggregate: dict,
                 candidate_label: str) -> None:
    lines = [
        "# Multi-document compact-hybrid confirmation",
        "",
        f"**{candidate_label}** was fixed before any captures from these documents were",
        "generated. Every predeclared document is retained below.",
        "",
        "| Document | Positions | Q2 KL | Candidate KL | KL recovered | JS recovered | Better chunks | Chunk interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['document']} | {row['positions']} | {row['baseline_kl']:.7f} | "
            f"{row['candidate_kl']:.7f} | {row['kl_recovery']:.2%} | "
            f"{row['js_recovery']:.2%} | {row['improved_chunks']}/{row['chunks']} | "
            f"[{row['chunk_interval_low']:.6f}, {row['chunk_interval_high']:.6f}] |")
    lines += [
        "",
        f"The candidate improves {aggregate['improved_documents']}/{aggregate['documents']} whole",
        f"documents. Pooled over {aggregate['positions']} positions, KL changes from",
        f"{aggregate['pooled_baseline_kl']:.7f} to {aggregate['pooled_candidate_kl']:.7f},",
        f"recovering {aggregate['pooled_kl_recovery']:.2%}; pooled JS recovery is",
        f"{aggregate['pooled_js_recovery']:.2%}.",
        "",
        f"Mean whole-document KL reduction is {aggregate['mean_document_kl_reduction']:.7f}",
        f"with descriptive four-document t interval [{aggregate['document_interval_low']:.7f},",
        f"{aggregate['document_interval_high']:.7f}]. This is the primary uncertainty summary;",
        "chunk intervals describe within-file consistency only.",
        "",
        "The documents are a small fixed set, not independent samples of all deployment",
        "workloads. Aggregate values are in",
        "`document_confirmation.csv`; chunk values are in `document_confirmation_chunks.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", nargs=4, action="append", default=[],
                        metavar=("LABEL", "REFERENCE", "BASELINE", "CANDIDATE"))
    parser.add_argument("--candidate-label", default="Q4 gate+up blocks 0-1")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_document_confirmation"))
    args = parser.parse_args()
    if len(args.document) < 2:
        parser.error("at least two --document entries are required")
    documents = [(label, Path(reference), Path(baseline), Path(candidate))
                 for label, reference, baseline, candidate in args.document]
    rows, chunks, aggregate = evaluate(documents)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "document_confirmation.csv", rows)
    write_csv(args.output_dir / "document_confirmation_chunks.csv", chunks)
    write_report(args.output_dir / "DOCUMENT_CONFIRMATION.md", rows, aggregate,
                 args.candidate_label)
    print(f"wrote {args.output_dir / 'DOCUMENT_CONFIRMATION.md'}")


if __name__ == "__main__":
    main()
