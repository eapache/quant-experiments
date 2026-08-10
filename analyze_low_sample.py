#!/usr/bin/env python3
"""Measure calibration stability as the number of whole context chunks changes."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from pathlib import Path

import numpy as np

from analyze_real_logits import (
    check_alignment,
    distribution_metrics,
    fit_temperature,
    fit_token_bias,
    read_kld,
)


def analyze_quant(reference_path: Path, candidate_path: Path, name: str,
                  temperatures: list[float], chunk_counts: list[int]) -> list[dict]:
    reference_file = read_kld(reference_path)
    candidate_file = read_kld(candidate_path)
    check_alignment(reference_file, candidate_file)
    reference = reference_file.decode()
    candidate = candidate_file.decode()
    reference_exact = reference_file.codes[:, : reference_file.n_vocab] > 0
    candidate_exact = candidate_file.codes[:, : candidate_file.n_vocab] > 0
    active = reference_exact | candidate_exact
    exact = reference_exact & candidate_exact
    chunks = reference_file.chunk_ids
    rows: list[dict] = []

    for fold in range(4):
        test_chunks = [2 * fold, 2 * fold + 1]
        test = np.isin(chunks, test_chunks)
        # Rotate the calibration order so every chunk participates at small sample sizes.
        order = [(2 * fold + 2 + offset) % reference_file.n_chunk
                 for offset in range(reference_file.n_chunk)]
        available = [chunk for chunk in order if chunk not in test_chunks]
        for chunk_count in chunk_counts:
            train_chunks = available[:chunk_count]
            train = np.isin(chunks, train_chunks)
            for temperature in temperatures:
                scale = fit_temperature(reference[train], candidate[train], active[train], temperature)
                scaled_train = scale * candidate[train]
                token_bias = fit_token_bias(
                    reference[train], candidate[train], exact[train], active[train], scaled_train)
                candidates = {
                    "identity": candidate[test],
                    "target-temperature": scale * candidate[test],
                    "target-temperature+token": scale * candidate[test] + token_bias,
                }
                for method, corrected in candidates.items():
                    metrics = distribution_metrics(reference[test], corrected, active[test], temperature)
                    rows.append({
                        "quant": name,
                        "fold": fold,
                        "test_chunks": "+".join(map(str, test_chunks)),
                        "train_chunks": "+".join(map(str, train_chunks)),
                        "calibration_chunks": chunk_count,
                        "calibration_positions": int(train.sum()),
                        "test_positions": int(test.sum()),
                        "temperature": temperature,
                        "method": method,
                        "scale": scale,
                        "equivalent_quant_temperature": temperature / scale,
                        **metrics,
                    })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Low-sample calibration stability",
        "",
        "Each outer fold holds out two complete chunks (126 positions). Calibration uses",
        "1, 2, 4, or 6 other complete chunks (63 to 378 positions). Scalar temperature",
        "is fitted directly at the downstream temperature shown, rather than fitted at T=1",
        "and transferred. Values are means across four outer folds.",
        "",
    ]
    methods = ["identity", "target-temperature", "target-temperature+token"]
    for quant in sorted({row["quant"] for row in rows}):
        for temperature in sorted({float(row["temperature"]) for row in rows}):
            lines += [f"## {quant} at T={temperature:g}", "",
                      "| Calibration positions | Method | KL | Recovered vs identity | Scale mean | Scale SD |",
                      "|---:|---|---:|---:|---:|---:|"]
            for positions in sorted({int(row["calibration_positions"]) for row in rows}):
                selected_base = [row for row in rows if row["quant"] == quant
                                 and float(row["temperature"]) == temperature
                                 and int(row["calibration_positions"]) == positions]
                identity = np.mean([float(row["kl"]) for row in selected_base if row["method"] == "identity"])
                scales = np.array([float(row["scale"]) for row in selected_base
                                   if row["method"] == "target-temperature"])
                for method in methods:
                    values = [float(row["kl"]) for row in selected_base if row["method"] == method]
                    mean = float(np.mean(values))
                    lines.append(f"| {positions} | {method} | {mean:.7f} | {1.0 - mean / identity:.2%} | {scales.mean():.6f} | {scales.std(ddof=1):.6f} |")
            lines.append("")
    lines += [
        "The repeated identity rows are intentional: they give the exactly matched held-out",
        "baseline for each fold/sample-size comparison. Fold-level results are in",
        "`low_sample.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("results/logits/q8.kld"))
    parser.add_argument("--q4", type=Path, default=Path("results/logits/q4.kld"))
    parser.add_argument("--q2", type=Path, default=Path("results/logits/q2.kld"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0])
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    jobs = (("Q4_K_XL", args.q4), ("Q2_K_XL", args.q2))
    rows: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(analyze_quant, args.reference, path, name,
                                   args.temperatures, args.chunk_counts) for name, path in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"finished {result[0]['quant']}", flush=True)
            rows.extend(result)
    rows.sort(key=lambda row: (row["quant"], row["fold"], row["calibration_positions"],
                               row["temperature"], row["method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "low_sample.csv", rows)
    write_report(args.output_dir / "LOW_SAMPLE.md", rows)
    print(f"wrote {args.output_dir / 'LOW_SAMPLE.md'}")


if __name__ == "__main__":
    main()
