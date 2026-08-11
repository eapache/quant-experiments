#!/usr/bin/env python3
"""Fit monotonic gap calibration on prose and evaluate it frozen on code."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np

from analyze_gap_calibration import evaluate, fit_gap_model
from analyze_sparse import fit_temperature, fit_token_bias, load_sparse


def evaluate_blocks(rows, chunks: np.ndarray, scale: float, bias: np.ndarray,
                    model, strength: float, blocks: int,
                    temperature: float) -> list[dict]:
    n_chunks = int(chunks.max()) + 1
    if n_chunks % blocks:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {blocks} blocks")
    block_size = n_chunks // blocks
    zero_bias = np.zeros_like(bias)
    output = []
    for block in range(blocks):
        block_chunks = list(range(block * block_size, (block + 1) * block_size))
        selected = np.isin(chunks, block_chunks)
        methods = {
            "identity": evaluate(rows, selected, temperature, 1.0, zero_bias),
            "frozen-target-temperature": evaluate(
                rows, selected, temperature, scale, zero_bias),
            "frozen-temperature+token": evaluate(
                rows, selected, temperature, scale, bias),
            "frozen-pairwise-gap": evaluate(
                rows, selected, temperature, scale, bias, model, strength, True),
        }
        for method, metrics in methods.items():
            output.append({
                "block": block, "chunks": "+".join(map(str, block_chunks)),
                "positions": int(selected.sum()), "method": method, "scale": scale,
                **metrics,
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], chunk_rows: list[dict],
                 reference_label: str, head: int, knots: int, rank_bins: int,
                 strength: float) -> None:
    identity = np.mean([row["kl"] for row in rows if row["method"] == "identity"])
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Frozen monotonic gap calibration on code", "",
        "The scalar temperature, static token bias, and pairwise monotonic mappings were",
        "fitted on all 2,016 prose positions and frozen before loading the code corpus.",
        f"The pairwise configuration is head={head}, knots={knots}, rank bins={rank_bins},",
        f"strength={strength:g}. It uses the modal discrete capacity and median accepted",
        "strength from the prose inner selections; no code position selected any setting.",
        f"The target is {reference_label} at T=1.", "",
        "| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        kl = np.mean([row["kl"] for row in selected])
        lines.append(
            f"| {method} | {kl:.7f} | {1.0 - kl / identity:.2%} | "
            f"{np.mean([row['js'] for row in selected]):.7f} | "
            f"{np.mean([row['tv'] for row in selected]):.7f} | "
            f"{np.mean([row['top1_agreement'] for row in selected]):.1%} | "
            f"{np.mean([row['top10_overlap'] for row in selected]):.1%} |")
    lines += ["", "## Per-chunk paired improvements over frozen temperature+token", "",
              "Positive values mean lower KL than the frozen static-bias baseline.", "",
              "| Method | Mean KL reduction | 95% interval | Improved chunks |",
              "|---|---:|---:|---:|"]
    baseline = {int(row["block"]): row["kl"] for row in chunk_rows
                if row["method"] == "frozen-temperature+token"}
    for method in methods:
        if method == "frozen-temperature+token":
            continue
        selected = [row for row in chunk_rows if row["method"] == method]
        differences = np.array([
            baseline[int(row["block"])] - row["kl"] for row in selected])
        half_width = 1.96 * differences.std(ddof=1) / np.sqrt(len(differences))
        lines.append(
            f"| {method} | {differences.mean():.7f} | "
            f"[{differences.mean() - half_width:.7f}, "
            f"{differences.mean() + half_width:.7f}] | "
            f"{np.count_nonzero(differences > 0)}/{len(differences)} |")
    lines += ["", "Block results are in `frozen_gap.csv`; per-chunk results are in",
              "`frozen_gap_chunks.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--train-candidate", type=Path, required=True)
    parser.add_argument("--ood-reference", type=Path, required=True)
    parser.add_argument("--ood-candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--knots", type=int, default=32)
    parser.add_argument("--rank-bins", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--stat-blocks", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_rows, _, n_vocab, _ = load_sparse(args.train_reference, args.train_candidate)
    train_selected = np.ones(len(train_rows), dtype=bool)
    scale = fit_temperature(train_rows, train_selected, args.temperature)
    bias = fit_token_bias(train_rows, train_selected, n_vocab, scale, args.temperature)
    model = fit_gap_model(train_rows, train_selected, scale, bias, args.temperature,
                          args.head, args.knots, args.rank_bins, True)
    del train_rows
    gc.collect()

    rows, chunks, ood_vocab, _ = load_sparse(args.ood_reference, args.ood_candidate)
    if ood_vocab != n_vocab:
        raise ValueError(f"unaligned vocabulary: {n_vocab} != {ood_vocab}")
    block_rows = evaluate_blocks(rows, chunks, scale, bias, model, args.strength,
                                 args.blocks, args.temperature)
    chunk_rows = evaluate_blocks(rows, chunks, scale, bias, model, args.strength,
                                 args.stat_blocks, args.temperature)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_gap.csv", block_rows)
    write_csv(args.output_dir / "frozen_gap_chunks.csv", chunk_rows)
    write_report(args.output_dir / "FROZEN_GAP.md", block_rows, chunk_rows,
                 args.reference_label, args.head, args.knots, args.rank_bins,
                 args.strength)
    print(f"wrote {args.output_dir / 'FROZEN_GAP.md'}")


if __name__ == "__main__":
    main()
