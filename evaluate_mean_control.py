#!/usr/bin/env python3
"""Evaluate fold-specific residual-stream control-vector logits."""

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


def evaluate(reference: Path, baseline: Path, candidates: list[Path],
             folds: int) -> list[dict]:
    baseline_rows, baseline_chunks, _, n_chunks = load_sparse(reference, baseline)
    if len(candidates) != folds:
        raise ValueError(f"expected {folds} candidate captures, got {len(candidates)}")
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    output = []
    for fold, candidate_path in enumerate(candidates):
        candidate_rows, candidate_chunks, _, candidate_n_chunks = load_sparse(
            reference, candidate_path)
        if candidate_n_chunks != n_chunks or not np.array_equal(candidate_chunks, baseline_chunks):
            raise ValueError(f"candidate capture {candidate_path} is not chunk-aligned")
        test_chunks = list(range(fold * fold_size, (fold + 1) * fold_size))
        selected = np.isin(baseline_chunks, test_chunks)
        for method, rows in (("identity", baseline_rows),
                             ("mean-layer-control", candidate_rows)):
            output.append({
                "fold": fold, "test_chunks": "+".join(map(str, test_chunks)),
                "method": method, **metrics(rows, selected, temperature=1.0),
            })
        print(f"finished fold {fold}", flush=True)
    return output


def write_report(path: Path, rows: list[dict], state_layer: int,
                 control_layer: int) -> None:
    methods = ["identity", "mean-layer-control"]
    means = {method: {metric: float(np.mean([row[metric] for row in rows
                                             if row["method"] == method]))
                      for metric in ("kl", "js", "tv", "top1_agreement", "top10_overlap")}
             for method in methods}
    raw_by_fold = {int(row["fold"]): row["kl"] for row in rows
                   if row["method"] == "identity"}
    corrected_by_fold = {int(row["fold"]): row["kl"] for row in rows
                         if row["method"] == "mean-layer-control"}
    reductions = np.array([raw_by_fold[fold] - corrected_by_fold[fold]
                           for fold in sorted(raw_by_fold)], dtype=np.float64)
    reduction = float(reductions.mean())
    half_width = float(3.182 * reductions.std(ddof=1) / np.sqrt(len(reductions)))
    raw = means["identity"]["kl"]
    lines = [
        "# Mean residual-stream control vector",
        "",
        f"Each outer fold fits one 2,560-float vector: the mean BF16-minus-Q2 input-state",
        f"difference at layer {state_layer}, using only the other 24 context chunks. llama.cpp",
        f"adds it after block {control_layer}, immediately before the measured target state.",
        "No BF16 output logits or held-out states are used to fit the sidecar, and strength is",
        "fixed at its natural value of 1.",
        "",
        "| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in (("identity", "Raw Q2"),
                          ("mean-layer-control", "Mean layer control")):
        value = means[method]
        lines.append(
            f"| {label} | {value['kl']:.7f} | {1.0 - value['kl'] / raw:.2%} | "
            f"{value['js']:.7f} | {value['tv']:.7f} | "
            f"{value['top1_agreement']:.1%} | {value['top10_overlap']:.1%} |")
    lines += [
        "",
        f"The control vector improves {int((reductions > 0).sum())}/{len(reductions)} outer",
        f"folds. Mean KL reduction is {reduction:.7f}; its four-block t interval is",
        f"[{reduction - half_width:.7f}, {reduction + half_width:.7f}].",
        "",
        "Fold metrics are in `mean_control_folds.csv`; vector metadata is in",
        "`control_vectors.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True,
                        help="candidate captures in fold order")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--state-layer", type=int, default=3)
    parser.add_argument("--control-layer", type=int, default=2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_mean_control_q2"))
    args = parser.parse_args()
    rows = evaluate(args.reference, args.baseline, args.candidates, args.folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mean_control_folds.csv", rows)
    write_report(args.output_dir / "MEAN_CONTROL.md", rows,
                 args.state_layer, args.control_layer)
    print(f"wrote {args.output_dir / 'MEAN_CONTROL.md'}")


if __name__ == "__main__":
    main()
