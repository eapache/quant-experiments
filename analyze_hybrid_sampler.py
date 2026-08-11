#!/usr/bin/env python3
"""Cross-validate temperature compensation stacked on a fixed hybrid model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_sampler_grid import prepare_reference, sampler_metrics
from analyze_sparse import load_sparse
from evaluate_document_confirmation import mean_t_interval


METHODS = ("same-settings", "q2-frozen-temperature", "hybrid-tuned-temperature")


def tune_temperature(rows, reference, orders, selected: np.ndarray,
                     top_p: float) -> tuple[float, float]:
    def score(temperature: float) -> float:
        return sampler_metrics(
            rows, reference, orders, selected, temperature, top_p)["sampler_js"]

    candidates = np.arange(0.70, 0.9001, 0.01)
    scored = [(score(float(temperature)), float(temperature))
              for temperature in candidates]
    _, center = min(scored)
    refined = np.arange(center - 0.01, center + 0.0101, 0.0025)
    return min((score(float(temperature)), float(temperature))
               for temperature in refined)


def analyze(reference_path: Path, candidate_path: Path, folds: int,
            calibration_positions: int, reference_temperature: float,
            reference_top_p: float, q2_temperature: float) -> tuple[list[dict], list[dict], dict]:
    rows, chunks, _, n_chunks = load_sparse(reference_path, candidate_path)
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be divided into {folds} folds")
    fold_size = n_chunks // folds
    reference = prepare_reference(rows, reference_temperature, reference_top_p)
    orders = [np.argsort(-row.candidate) for row in rows]
    fold_rows = []
    chunk_rows = []
    selected_temperatures = []
    for fold in range(folds):
        test_chunks = list(range(fold * fold_size, (fold + 1) * fold_size))
        test = np.isin(chunks, test_chunks)
        train_indices = np.flatnonzero(~test)
        chosen = train_indices[np.linspace(
            0, len(train_indices) - 1,
            min(calibration_positions, len(train_indices)), dtype=int)]
        train = np.zeros(len(rows), dtype=bool)
        train[chosen] = True
        training_js, tuned_temperature = tune_temperature(
            rows, reference, orders, train, reference_top_p)
        selected_temperatures.append(tuned_temperature)
        settings = {
            "same-settings": reference_temperature,
            "q2-frozen-temperature": q2_temperature,
            "hybrid-tuned-temperature": tuned_temperature,
        }
        for method, temperature in settings.items():
            result = sampler_metrics(
                rows, reference, orders, test, temperature, reference_top_p)
            fold_rows.append({
                "fold": fold,
                "test_chunks": "+".join(map(str, test_chunks)),
                "calibration_positions": int(train.sum()),
                "test_positions": int(test.sum()),
                "method": method,
                "candidate_temperature": temperature,
                "candidate_top_p": reference_top_p,
                "training_js": training_js if method == "hybrid-tuned-temperature" else "",
                **result,
            })
        for chunk in test_chunks:
            chunk_selected = chunks == chunk
            for method, temperature in settings.items():
                result = sampler_metrics(
                    rows, reference, orders, chunk_selected, temperature, reference_top_p)
                chunk_rows.append({
                    "chunk": chunk,
                    "outer_fold": fold,
                    "positions": int(chunk_selected.sum()),
                    "method": method,
                    "candidate_temperature": temperature,
                    "candidate_top_p": reference_top_p,
                    **result,
                })

    chunk_maps = {
        method: {row["chunk"]: row["sampler_js"] for row in chunk_rows
                 if row["method"] == method}
        for method in METHODS
    }
    baseline = chunk_maps["same-settings"]
    q2_frozen = chunk_maps["q2-frozen-temperature"]
    tuned = chunk_maps["hybrid-tuned-temperature"]
    chunks_order = sorted(baseline)
    q2_gains = np.array([baseline[c] - q2_frozen[c] for c in chunks_order])
    tuned_gains = np.array([baseline[c] - tuned[c] for c in chunks_order])
    incremental = np.array([q2_frozen[c] - tuned[c] for c in chunks_order])
    q2_mean, q2_low, q2_high = mean_t_interval(q2_gains)
    tuned_mean, tuned_low, tuned_high = mean_t_interval(tuned_gains)
    inc_mean, inc_low, inc_high = mean_t_interval(incremental)
    summary = {
        "frozen_hybrid_temperature": float(np.mean(selected_temperatures)),
        "fold_temperatures": "+".join(f"{value:.4f}" for value in selected_temperatures),
        "q2_mean_js_reduction": q2_mean,
        "q2_interval_low": q2_low,
        "q2_interval_high": q2_high,
        "q2_improved_chunks": int((q2_gains > 0).sum()),
        "tuned_mean_js_reduction": tuned_mean,
        "tuned_interval_low": tuned_low,
        "tuned_interval_high": tuned_high,
        "tuned_improved_chunks": int((tuned_gains > 0).sum()),
        "incremental_mean_js_reduction": inc_mean,
        "incremental_interval_low": inc_low,
        "incremental_interval_high": inc_high,
        "incrementally_improved_chunks": int((incremental > 0).sum()),
        "passes_hybrid_specific_gate": inc_low > 0,
    }
    return fold_rows, chunk_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, fold_rows: list[dict], summary: dict,
                 reference_temperature: float, reference_top_p: float,
                 q2_temperature: float) -> None:
    lines = [
        "# Temperature compensation stacked on compact hybrid",
        "",
        f"The target is BF16 at T={reference_temperature:.2f}/top-p={reference_top_p:.2f}.",
        f"Four outer folds compare identical hybrid settings, the previously frozen Q2",
        f"temperature T={q2_temperature:.4f}, and temperature refit only on other prose chunks.",
        "A hybrid-specific setting is promoted only if it beats the Q2-frozen temperature",
        "with a paired 32-chunk interval above zero.",
        "",
        "| Method | JS | Recovery vs same settings | TV | Candidate T |",
        "|---|---:|---:|---:|---:|",
    ]
    identity = np.mean([row["sampler_js"] for row in fold_rows
                        if row["method"] == "same-settings"])
    for method in METHODS:
        selected = [row for row in fold_rows if row["method"] == method]
        js = float(np.mean([row["sampler_js"] for row in selected]))
        tv = float(np.mean([row["sampler_tv"] for row in selected]))
        temperature = float(np.mean([row["candidate_temperature"] for row in selected]))
        lines.append(
            f"| {method} | {js:.7f} | {1.0 - js / identity:.2%} | {tv:.7f} | "
            f"{temperature:.4f} |")
    verdict = "passes" if summary["passes_hybrid_specific_gate"] else "fails"
    lines += [
        "",
        f"Nested hybrid temperatures are {summary['fold_temperatures']} (mean",
        f"{summary['frozen_hybrid_temperature']:.4f}). The hybrid-specific setting improves",
        f"{summary['incrementally_improved_chunks']}/32 chunks over Q2's frozen temperature.",
        f"Mean incremental JS reduction is {summary['incremental_mean_js_reduction']:.7f},",
        f"with interval [{summary['incremental_interval_low']:.7f},",
        f"{summary['incremental_interval_high']:.7f}]. It **{verdict}** the promotion gate.",
        "",
        "Fold results are in `hybrid_sampler.csv`; paired chunk results are in",
        "`hybrid_sampler_chunks.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--calibration-positions", type=int, default=256)
    parser.add_argument("--reference-temperature", type=float, default=0.8)
    parser.add_argument("--reference-top-p", type=float, default=0.95)
    parser.add_argument("--q2-temperature", type=float, default=0.7625)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_hybrid_sampler"))
    args = parser.parse_args()
    fold_rows, chunk_rows, summary = analyze(
        args.reference, args.candidate, args.folds, args.calibration_positions,
        args.reference_temperature, args.reference_top_p, args.q2_temperature)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "hybrid_sampler.csv", fold_rows)
    write_csv(args.output_dir / "hybrid_sampler_chunks.csv", chunk_rows)
    write_report(args.output_dir / "HYBRID_SAMPLER.md", fold_rows, summary,
                 args.reference_temperature, args.reference_top_p, args.q2_temperature)
    print(f"wrote {args.output_dir / 'HYBRID_SAMPLER.md'}")


if __name__ == "__main__":
    main()
