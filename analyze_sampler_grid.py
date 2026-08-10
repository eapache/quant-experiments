#!/usr/bin/env python3
"""Cross-validate ordinary temperature/top-p compensation against a reference sampler."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from pathlib import Path

import numpy as np

from analyze_sparse import PairedRow, load_sparse, softmax


def nucleus_with_order(probabilities: np.ndarray, order: np.ndarray, top_p: float) -> np.ndarray:
    sorted_p = probabilities[order]
    keep = np.cumsum(sorted_p) - sorted_p < top_p
    sorted_p = np.where(keep, sorted_p, 0.0)
    sorted_p /= sorted_p.sum()
    result = np.zeros_like(probabilities)
    result[order] = sorted_p
    return result


def prepare_reference(rows: list[PairedRow], temperature: float, top_p: float) -> list[np.ndarray]:
    result = []
    for row in rows:
        p = softmax(row.reference / temperature)
        order = np.argsort(-row.reference)
        result.append(nucleus_with_order(p, order, top_p))
    return result


def sampler_metrics(rows: list[PairedRow], reference: list[np.ndarray], candidate_orders: list[np.ndarray],
                    selected: np.ndarray, temperature: float, top_p: float) -> dict[str, float]:
    js = []
    tv = []
    support = []
    for index in np.flatnonzero(selected):
        row = rows[index]
        p = reference[index]
        q = softmax(row.candidate / temperature)
        q = nucleus_with_order(q, candidate_orders[index], top_p)
        midpoint = 0.5 * (p + q)
        p_keep = p > 0
        q_keep = q > 0
        js.append(float(0.5 * np.dot(p[p_keep], np.log(p[p_keep] / midpoint[p_keep]))
                        + 0.5 * np.dot(q[q_keep], np.log(q[q_keep] / midpoint[q_keep]))))
        tv.append(float(0.5 * np.abs(p - q).sum()))
        support.append(float(np.count_nonzero(p_keep & q_keep) /
                             max(1, np.count_nonzero(p_keep | q_keep))))
    return {
        "sampler_js": float(np.mean(js)),
        "sampler_tv": float(np.mean(tv)),
        "sampler_support_jaccard": float(np.mean(support)),
    }


def search(rows: list[PairedRow], reference: list[np.ndarray], candidate_orders: list[np.ndarray],
           selected: np.ndarray, reference_temperature: float,
           reference_top_p: float) -> dict[str, tuple[float, float, float]]:
    def score(temperature: float, top_p: float) -> float:
        return sampler_metrics(
            rows, reference, candidate_orders, selected, temperature, top_p)["sampler_js"]

    coarse_temperatures = np.arange(
        reference_temperature - 0.10, reference_temperature + 0.1001, 0.025)
    best_temperature = (float("inf"), reference_temperature, reference_top_p)
    for temperature in coarse_temperatures:
        value = score(float(temperature), reference_top_p)
        if value < best_temperature[0]:
            best_temperature = (value, float(temperature), reference_top_p)

    center_temperature = best_temperature[1]
    for temperature in np.arange(center_temperature - 0.02, center_temperature + 0.0201, 0.005):
        value = score(float(temperature), reference_top_p)
        if value < best_temperature[0]:
            best_temperature = (value, float(temperature), reference_top_p)

    top_ps = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    best_top_p = (float("inf"), reference_temperature, reference_top_p)
    for top_p in top_ps:
        value = score(reference_temperature, top_p)
        if value < best_top_p[0]:
            best_top_p = (value, reference_temperature, top_p)

    center_top_p = best_top_p[2]
    for top_p in np.arange(max(0.88, center_top_p - 0.015), min(0.995, center_top_p + 0.0151), 0.005):
        value = score(reference_temperature, float(top_p))
        if value < best_top_p[0]:
            best_top_p = (value, reference_temperature, float(top_p))

    best = (float("inf"), reference_temperature, reference_top_p)
    for temperature in np.arange(reference_temperature - 0.10,
                                 reference_temperature + 0.1001, 0.01):
        for top_p in np.arange(max(0.88, reference_top_p - 0.05),
                               min(0.995, reference_top_p + 0.0401), 0.01):
            value = score(float(temperature), float(top_p))
            if value < best[0]:
                best = (value, float(temperature), float(top_p))

    center_temperature = best[1]
    center_top_p = best[2]
    for temperature in np.arange(center_temperature - 0.01, center_temperature + 0.0101, 0.005):
        for top_p in np.arange(max(0.88, center_top_p - 0.01),
                               min(0.995, center_top_p + 0.0101), 0.005):
            value = score(float(temperature), float(top_p))
            if value < best[0]:
                best = (value, float(temperature), float(top_p))
    return {
        "tuned-temperature": best_temperature,
        "tuned-top-p": best_top_p,
        "tuned-temperature-top-p": best,
    }


def analyze_quant(reference_path: Path, candidate_path: Path, name: str, folds: int,
                  calibration_positions: int, reference_temperature: float,
                  reference_top_p: float) -> list[dict]:
    rows, chunks, _, n_chunks = load_sparse(reference_path, candidate_path)
    fold_size = n_chunks // folds
    reference = prepare_reference(rows, reference_temperature, reference_top_p)
    candidate_orders = [np.argsort(-row.candidate) for row in rows]
    output = []
    for fold in range(folds):
        test_chunks = list(range(fold * fold_size, (fold + 1) * fold_size))
        test = np.isin(chunks, test_chunks)
        train_indices = np.flatnonzero(~test)
        chosen = train_indices[np.linspace(0, len(train_indices) - 1,
                                           min(calibration_positions, len(train_indices)), dtype=int)]
        train = np.zeros(len(rows), dtype=bool)
        train[chosen] = True
        tuned = search(
            rows, reference, candidate_orders, train, reference_temperature, reference_top_p)
        configurations = {
            "same-settings": (reference_temperature, reference_top_p, ""),
        }
        configurations.update({method: (temperature, top_p, train_js)
                               for method, (train_js, temperature, top_p) in tuned.items()})
        for method, (candidate_temperature, candidate_top_p, train_js) in configurations.items():
            result = sampler_metrics(
                rows, reference, candidate_orders, test, candidate_temperature, candidate_top_p)
            output.append({
                "quant": name,
                "fold": fold,
                "test_chunks": "+".join(map(str, test_chunks)),
                "calibration_positions": int(train.sum()),
                "test_positions": int(test.sum()),
                "method": method,
                "reference_temperature": reference_temperature,
                "reference_top_p": reference_top_p,
                "candidate_temperature": candidate_temperature,
                "candidate_top_p": candidate_top_p,
                "training_js": train_js,
                **result,
            })
    return output


def write_report(path: Path, rows: list[dict], reference_label: str) -> None:
    calibration_positions = sorted({int(row["calibration_positions"]) for row in rows})
    calibration_text = ", ".join(map(str, calibration_positions))
    lines = [
        "# Held-out ordinary sampler compensation",
        "",
        f"For each outer fold, {calibration_text} evenly spaced positions from the other chunks tune only",
        f"the candidate quant's temperature and top-p. The target is the {reference_label} distribution at",
        "T=0.8/top-p=0.95. All reported divergences are on complete held-out chunk blocks.",
        "",
        "| Quant | Method | JS | Recovered | TV | Support Jaccard | Candidate T | Candidate top-p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for quant in sorted({row["quant"] for row in rows}):
        selected = [row for row in rows if row["quant"] == quant]
        identity = np.mean([row["sampler_js"] for row in selected if row["method"] == "same-settings"])
        for method in ("same-settings", "tuned-temperature", "tuned-top-p", "tuned-temperature-top-p"):
            values = [row for row in selected if row["method"] == method]
            js = np.mean([row["sampler_js"] for row in values])
            tv = np.mean([row["sampler_tv"] for row in values])
            support = np.mean([row["sampler_support_jaccard"] for row in values])
            temperature = np.mean([row["candidate_temperature"] for row in values])
            top_p = np.mean([row["candidate_top_p"] for row in values])
            lines.append(f"| {quant} | {method} | {js:.7f} | {1.0 - js / identity:.2%} | {tv:.7f} | {support:.1%} | {temperature:.4f} | {top_p:.4f} |")
    lines += ["", "## Fold-level tuned parameters", "",
              "| Quant | Fold | Candidate T | Candidate top-p | Train JS | Test JS |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["method"] == "tuned-temperature-top-p":
            lines.append(f"| {row['quant']} | {row['fold']} | {row['candidate_temperature']:.4f} | {row['candidate_top_p']:.4f} | {row['training_js']:.7f} | {row['sampler_js']:.7f} |")
    lines += ["", "Raw fold-level results are in `sampler_grid.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--q8", type=Path)
    parser.add_argument("--q4", type=Path)
    parser.add_argument("--q2", type=Path)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--calibration-positions", type=int, default=256)
    parser.add_argument("--reference-temperature", type=float, default=0.8)
    parser.add_argument("--reference-top-p", type=float, default=0.95)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/gpu32"))
    args = parser.parse_args()
    jobs = tuple((name, path) for name, path in (
        ("Q8_K_XL", args.q8), ("Q4_K_XL", args.q4), ("Q2_K_XL", args.q2)
    ) if path is not None)
    if not jobs:
        parser.error("at least one of --q8, --q4, or --q2 is required")
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(
            analyze_quant, args.reference, path, name, args.folds,
            args.calibration_positions, args.reference_temperature, args.reference_top_p)
            for name, path in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"finished {result[0]['quant']}", flush=True)
            rows.extend(result)
    rows.sort(key=lambda row: (row["quant"], row["fold"], row["method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampler_grid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir / "SAMPLER.md", rows, args.reference_label)
    print(f"wrote {args.output_dir / 'SAMPLER.md'}")


if __name__ == "__main__":
    main()
