#!/usr/bin/env python3
"""Evaluate sampler settings selected on one corpus without refitting on another."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_sampler_grid import prepare_reference, sampler_metrics
from analyze_sparse import load_sparse


SETTINGS = {
    "Q2_K_XL": {
        "same-settings": (0.8, 0.95),
        "frozen-temperature": (0.76, 0.95),
        "frozen-top-p": (0.8, 0.93125),
        "frozen-joint": (0.7275, 0.965),
    },
    "Q4_K_XL": {
        "same-settings": (0.8, 0.95),
        "frozen-temperature": (0.795, 0.95),
        "frozen-joint": (0.78125, 0.9575),
    },
}


def evaluate(reference_path: Path, candidate_path: Path, quant: str, blocks: int) -> list[dict]:
    rows, chunks, _, n_chunks = load_sparse(reference_path, candidate_path)
    reference = prepare_reference(rows, 0.8, 0.95)
    orders = [np.argsort(-row.candidate) for row in rows]
    block_size = n_chunks // blocks
    output = []
    for block in range(blocks):
        block_chunks = list(range(block * block_size, (block + 1) * block_size))
        selected = np.isin(chunks, block_chunks)
        for method, (temperature, top_p) in SETTINGS[quant].items():
            result = sampler_metrics(rows, reference, orders, selected, temperature, top_p)
            output.append({
                "quant": quant,
                "block": block,
                "chunks": "+".join(map(str, block_chunks)),
                "positions": int(selected.sum()),
                "method": method,
                "candidate_temperature": temperature,
                "candidate_top_p": top_p,
                **result,
            })
    return output


def write_report(path: Path, rows: list[dict], chunk_rows: list[dict]) -> None:
    block_count = len({int(row["block"]) for row in rows})
    lines = [
        "# Frozen sampler settings on an out-of-domain code corpus",
        "",
        "Settings were selected on `chats/first.txt` and frozen before evaluating the Python",
        "source of the synthetic lab. No code-corpus positions were used for fitting. Q8 uses",
        f"T=0.8/top-p=0.95. Values are means across {block_count} held-out code blocks.",
        "",
        "| Quant | Method | T | top-p | JS | Block SD | Recovered | TV |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for quant in sorted({row["quant"] for row in rows}):
        selected = [row for row in rows if row["quant"] == quant]
        baseline = np.mean([row["sampler_js"] for row in selected
                            if row["method"] == "same-settings"])
        for method in SETTINGS[quant]:
            values = [row for row in selected if row["method"] == method]
            js = np.array([row["sampler_js"] for row in values])
            tv = np.mean([row["sampler_tv"] for row in values])
            lines.append(f"| {quant} | {method} | {values[0]['candidate_temperature']:.4f} | {values[0]['candidate_top_p']:.4f} | {js.mean():.7f} | {js.std(ddof=1):.7f} | {1.0 - js.mean() / baseline:.2%} | {tv:.7f} |")
    lines += ["", "## Per-chunk paired improvements", "",
              "Positive values mean lower JS than the same-settings baseline. The interval is",
              "a normal 95% interval over 32 complete 63-position chunks.", "",
              "| Quant | Method | Mean JS reduction | 95% interval | Improved chunks |",
              "|---|---|---:|---:|---:|"]
    for quant in sorted({row["quant"] for row in chunk_rows}):
        selected = [row for row in chunk_rows if row["quant"] == quant]
        baseline = {int(row["block"]): float(row["sampler_js"]) for row in selected
                    if row["method"] == "same-settings"}
        for method in SETTINGS[quant]:
            if method == "same-settings":
                continue
            values = [row for row in selected if row["method"] == method]
            differences = np.array([
                baseline[int(row["block"])] - float(row["sampler_js"]) for row in values])
            half_width = 1.96 * differences.std(ddof=1) / np.sqrt(len(differences))
            lines.append(f"| {quant} | {method} | {differences.mean():.7f} | [{differences.mean() - half_width:.7f}, {differences.mean() + half_width:.7f}] | {np.count_nonzero(differences > 0)}/{len(differences)} |")
    lines += ["", "Raw block-level results are in `frozen_sampler.csv`; per-chunk values are in",
              "`frozen_sampler_chunks.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--stat-blocks", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ood_code"))
    args = parser.parse_args()
    rows = []
    rows.extend(evaluate(args.reference, args.q4, "Q4_K_XL", args.blocks))
    rows.extend(evaluate(args.reference, args.q2, "Q2_K_XL", args.blocks))
    rows.sort(key=lambda row: (row["quant"], row["block"], row["method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frozen_sampler.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    chunk_rows = []
    chunk_rows.extend(evaluate(args.reference, args.q4, "Q4_K_XL", args.stat_blocks))
    chunk_rows.extend(evaluate(args.reference, args.q2, "Q2_K_XL", args.stat_blocks))
    chunk_rows.sort(key=lambda row: (row["quant"], row["block"], row["method"]))
    with (args.output_dir / "frozen_sampler_chunks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(chunk_rows[0]))
        writer.writeheader()
        writer.writerows(chunk_rows)
    write_report(args.output_dir / "SAMPLER.md", rows, chunk_rows)
    print(f"wrote {args.output_dir / 'SAMPLER.md'}")


if __name__ == "__main__":
    main()
