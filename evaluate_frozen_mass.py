#!/usr/bin/env python3
"""Freeze prose-fitted cumulative-mass calibration and evaluate it on code."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_mass_calibration import (baseline_metrics, calibrated_metrics,
                                      fit_mass_model, read_sampler_settings)
from analyze_sampler_grid import prepare_reference
from analyze_sparse import load_sparse


def averaged_settings(path: Path, quant: str) -> dict[str, tuple[float, float]]:
    by_fold = read_sampler_settings(path, quant)
    methods = sorted({method for settings in by_fold.values() for method in settings})
    result = {}
    for method in methods:
        values = [settings[method] for settings in by_fold.values() if method in settings]
        result[method] = tuple(np.mean(values, axis=0))
    return result


def evaluate_blocks(rows, chunks: np.ndarray, reference: list[np.ndarray], model,
                    settings: dict[str, tuple[float, float]], blocks: int,
                    reference_temperature: float, reference_top_p: float) -> list[dict]:
    orders = [np.argsort(-row.candidate) for row in rows]
    n_chunks = int(chunks.max()) + 1
    block_size = n_chunks // blocks
    output = []
    for block in range(blocks):
        block_chunks = list(range(block * block_size, (block + 1) * block_size))
        selected = np.isin(chunks, block_chunks)
        methods = {
            "same-settings": baseline_metrics(
                rows, reference, orders, selected, reference_temperature,
                reference_top_p, reference_temperature),
            "frozen-mass-global": calibrated_metrics(
                rows, reference, selected, model, reference_temperature,
                reference_temperature),
        }
        for source, output_name in (
            ("tuned-temperature", "frozen-temperature"),
            ("tuned-top-p", "frozen-top-p"),
            ("tuned-temperature-top-p", "frozen-joint"),
        ):
            if source in settings:
                temperature, top_p = settings[source]
                methods[output_name] = baseline_metrics(
                    rows, reference, orders, selected, temperature, top_p,
                    reference_temperature)
        for method, metrics in methods.items():
            output.append({
                "block": block, "chunks": "+".join(map(str, block_chunks)),
                "positions": int(selected.sum()), "method": method, **metrics,
            })
    return output


def write_report(path: Path, rows: list[dict], chunk_rows: list[dict],
                 model_top_p: float, reference_label: str) -> None:
    baseline = np.mean([row["sampler_js"] for row in rows
                        if row["method"] == "same-settings"])
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Frozen cumulative-mass calibration on code", "",
        "A single cumulative-mass curve was fitted on all 2,016 prose positions and frozen",
        f"before reading the code corpus. It implied candidate top-p={model_top_p:.6f} at T=0.8",
        f"for the {reference_label} T=0.8/top-p=0.95 target. Scalar baselines are the fold-mean",
        "settings selected by the earlier prose sampler search.", "",
        "| Method | JS | Recovered | TV | Support Jaccard | Ref mass retained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        js = np.mean([row["sampler_js"] for row in selected])
        lines.append(
            f"| {method} | {js:.7f} | {1.0 - js / baseline:.2%} | "
            f"{np.mean([row['sampler_tv'] for row in selected]):.7f} | "
            f"{np.mean([row['sampler_support_jaccard'] for row in selected]):.1%} | "
            f"{np.mean([row['reference_mass_in_candidate_support'] for row in selected]):.2%} |")
    lines += ["", "## Per-chunk paired improvements", "",
              "Positive values mean lower JS than same settings.", "",
              "| Method | Mean JS reduction | 95% interval | Improved chunks |",
              "|---|---:|---:|---:|"]
    chunk_baseline = {int(row["block"]): row["sampler_js"] for row in chunk_rows
                      if row["method"] == "same-settings"}
    for method in methods:
        if method == "same-settings":
            continue
        selected = [row for row in chunk_rows if row["method"] == method]
        differences = np.array([
            chunk_baseline[int(row["block"])] - row["sampler_js"] for row in selected])
        half_width = 1.96 * differences.std(ddof=1) / np.sqrt(len(differences))
        lines.append(
            f"| {method} | {differences.mean():.7f} | "
            f"[{differences.mean() - half_width:.7f}, "
            f"{differences.mean() + half_width:.7f}] | "
            f"{np.count_nonzero(differences > 0)}/{len(differences)} |")
    lines += ["", "Block results are in `frozen_mass.csv`; per-chunk results are in",
              "`frozen_mass_chunks.csv`.", ""]
    path.write_text("\n".join(lines))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--train-candidate", type=Path, required=True)
    parser.add_argument("--ood-reference", type=Path, required=True)
    parser.add_argument("--ood-candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--sampler-settings-csv", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--stat-blocks", type=int, default=32)
    parser.add_argument("--mass-points", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_rows, _, _, _ = load_sparse(args.train_reference, args.train_candidate)
    train_selected = np.ones(len(train_rows), dtype=bool)
    mass_grid = np.linspace(0.80, 0.9995, args.mass_points)
    model = fit_mass_model(train_rows, train_selected, 1, 0.8, 0.8, 0.95, mass_grid)
    settings = averaged_settings(args.sampler_settings_csv, args.candidate_name)
    rows, chunks, _, _ = load_sparse(args.ood_reference, args.ood_candidate)
    reference = prepare_reference(rows, 0.8, 0.95)
    block_rows = evaluate_blocks(
        rows, chunks, reference, model, settings, args.blocks, 0.8, 0.95)
    chunk_rows = evaluate_blocks(
        rows, chunks, reference, model, settings, args.stat_blocks, 0.8, 0.95)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_mass.csv", block_rows)
    write_csv(args.output_dir / "frozen_mass_chunks.csv", chunk_rows)
    write_report(args.output_dir / "FROZEN_MASS.md", block_rows, chunk_rows,
                 float(model.candidate_top_ps[0]), args.reference_label)
    print(f"wrote {args.output_dir / 'FROZEN_MASS.md'}")


if __name__ == "__main__":
    main()
