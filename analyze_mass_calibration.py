#!/usr/bin/env python3
"""Cross-validate cumulative-mass calibration for a quantized nucleus sampler."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_sampler_grid import (nucleus_with_order, prepare_reference,
                                  sampler_metrics)
from analyze_sparse import PairedRow, load_sparse, softmax


@dataclass
class MassModel:
    entropy_thresholds: np.ndarray
    candidate_top_ps: np.ndarray


def candidate_entropy(row: PairedRow, temperature: float) -> float:
    q = softmax(row.candidate / temperature)
    return float(-np.dot(q, np.log(q)))


def row_mass_curve(row: PairedRow, candidate_temperature: float,
                   reference_temperature: float, mass_grid: np.ndarray) -> np.ndarray:
    q = softmax(row.candidate / candidate_temperature)
    p = softmax(row.reference / reference_temperature)
    order = np.argsort(-row.candidate)
    q_cumulative = np.cumsum(q[order], dtype=np.float64)
    p_cumulative = np.cumsum(p[order], dtype=np.float64)
    indices = np.searchsorted(q_cumulative, mass_grid, side="left")
    indices = np.minimum(indices, len(order) - 1)
    return p_cumulative[indices]


def fit_mass_model(rows: list[PairedRow], selected: np.ndarray, groups: int,
                   candidate_temperature: float, reference_temperature: float,
                   reference_top_p: float, mass_grid: np.ndarray) -> MassModel:
    indices = np.flatnonzero(selected)
    entropies = np.array([
        candidate_entropy(rows[index], candidate_temperature) for index in indices])
    entropy_thresholds = (np.quantile(entropies, np.arange(1, groups) / groups)
                          if groups > 1 else np.empty(0))
    assignments = np.searchsorted(entropy_thresholds, entropies, side="right")
    curves = np.stack([
        row_mass_curve(rows[index], candidate_temperature, reference_temperature, mass_grid)
        for index in indices
    ])
    candidate_top_ps = np.empty(groups, dtype=np.float64)
    global_curve = np.maximum.accumulate(curves.mean(axis=0))
    for group in range(groups):
        group_curves = curves[assignments == group]
        mean_curve = (np.maximum.accumulate(group_curves.mean(axis=0))
                      if len(group_curves) else global_curve)
        candidate_top_ps[group] = np.interp(
            reference_top_p, mean_curve, mass_grid,
            left=mass_grid[0], right=mass_grid[-1])
    return MassModel(entropy_thresholds.astype(np.float32),
                     candidate_top_ps.astype(np.float32))


def calibrated_metrics(rows: list[PairedRow], reference: list[np.ndarray],
                       selected: np.ndarray, model: MassModel,
                       candidate_temperature: float,
                       reference_temperature: float) -> dict[str, float]:
    js, tv, support, retained, predicted_top_p = [], [], [], [], []
    for index in np.flatnonzero(selected):
        row = rows[index]
        p = reference[index]
        q_full = softmax(row.candidate / candidate_temperature)
        entropy = -float(np.dot(q_full, np.log(q_full)))
        group = int(np.searchsorted(model.entropy_thresholds, entropy, side="right"))
        top_p = float(model.candidate_top_ps[group])
        order = np.argsort(-row.candidate)
        q = nucleus_with_order(q_full, order, top_p)
        midpoint = 0.5 * (p + q)
        p_keep, q_keep = p > 0, q > 0
        js.append(float(0.5 * np.dot(p[p_keep], np.log(p[p_keep] / midpoint[p_keep]))
                        + 0.5 * np.dot(q[q_keep], np.log(q[q_keep] / midpoint[q_keep]))))
        tv.append(float(0.5 * np.abs(p - q).sum()))
        support.append(float(np.count_nonzero(p_keep & q_keep)
                             / max(1, np.count_nonzero(p_keep | q_keep))))
        retained.append(float(np.dot(
            softmax(row.reference / reference_temperature), q_keep)))
        predicted_top_p.append(top_p)
    return {
        "sampler_js": float(np.mean(js)),
        "sampler_tv": float(np.mean(tv)),
        "sampler_support_jaccard": float(np.mean(support)),
        "reference_mass_in_candidate_support": float(np.mean(retained)),
        "mean_candidate_top_p": float(np.mean(predicted_top_p)),
    }


def baseline_metrics(rows: list[PairedRow], reference: list[np.ndarray],
                     orders: list[np.ndarray], selected: np.ndarray,
                     temperature: float, top_p: float,
                     reference_temperature: float) -> dict[str, float]:
    metrics = sampler_metrics(rows, reference, orders, selected, temperature, top_p)
    retained = []
    for index in np.flatnonzero(selected):
        q = softmax(rows[index].candidate / temperature)
        nucleus = nucleus_with_order(q, orders[index], top_p)
        retained.append(float(np.dot(
            softmax(rows[index].reference / reference_temperature), nucleus > 0)))
    return {
        **metrics,
        "reference_mass_in_candidate_support": float(np.mean(retained)),
        "mean_candidate_top_p": top_p,
    }


def select_groups(rows: list[PairedRow], reference: list[np.ndarray],
                  inner_train: np.ndarray, inner_validation: np.ndarray,
                  groups_grid: list[int], candidate_temperature: float,
                  reference_temperature: float, reference_top_p: float,
                  mass_grid: np.ndarray) -> tuple[int, bool]:
    global_model = fit_mass_model(
        rows, inner_train, 1, candidate_temperature, reference_temperature,
        reference_top_p, mass_grid)
    global_js = calibrated_metrics(
        rows, reference, inner_validation, global_model, candidate_temperature,
        reference_temperature)["sampler_js"]
    best = (float("inf"), groups_grid[0])
    for groups in groups_grid:
        model = fit_mass_model(
            rows, inner_train, groups, candidate_temperature, reference_temperature,
            reference_top_p, mass_grid)
        js = calibrated_metrics(
            rows, reference, inner_validation, model, candidate_temperature,
            reference_temperature)["sampler_js"]
        best = min(best, (js, groups))
    return int(best[1]), bool(best[0] < global_js)


def read_sampler_settings(path: Path | None, quant: str) -> dict[int, dict[str, tuple[float, float]]]:
    if path is None:
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[int, dict[str, tuple[float, float]]] = {}
    for row in rows:
        if row["quant"] != quant:
            continue
        result.setdefault(int(row["fold"]), {})[row["method"]] = (
            float(row["candidate_temperature"]), float(row["candidate_top_p"]))
    return result


def analyze(reference_path: Path, candidate_path: Path, name: str, folds: int,
            reference_temperature: float, reference_top_p: float,
            candidate_temperature: float, groups_grid: list[int], mass_points: int,
            sampler_settings_path: Path | None) -> list[dict]:
    rows, chunks, _, n_chunks = load_sparse(reference_path, candidate_path)
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    reference = prepare_reference(rows, reference_temperature, reference_top_p)
    orders = [np.argsort(-row.candidate) for row in rows]
    mass_grid = np.linspace(0.80, 0.9995, mass_points)
    settings = read_sampler_settings(sampler_settings_path, name)
    fold_size = n_chunks // folds
    output: list[dict] = []
    for fold in range(folds):
        test_chunks = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_chunks = [chunk for chunk in range(n_chunks) if chunk not in test_chunks]
        inner_validation_chunks = train_chunks[fold % 4::4]
        inner_train_chunks = [chunk for chunk in train_chunks
                              if chunk not in inner_validation_chunks]
        test = np.isin(chunks, test_chunks)
        train = np.isin(chunks, train_chunks)
        inner_train = np.isin(chunks, inner_train_chunks)
        inner_validation = np.isin(chunks, inner_validation_chunks)
        groups, enabled = select_groups(
            rows, reference, inner_train, inner_validation, groups_grid,
            candidate_temperature, reference_temperature, reference_top_p, mass_grid)
        global_model = fit_mass_model(
            rows, train, 1, candidate_temperature, reference_temperature,
            reference_top_p, mass_grid)
        conditioned_model = fit_mass_model(
            rows, train, groups, candidate_temperature, reference_temperature,
            reference_top_p, mass_grid)
        methods: dict[str, tuple[dict[str, float], int, MassModel | None, bool, float]] = {
            "same-settings": (
                baseline_metrics(rows, reference, orders, test,
                                 reference_temperature, reference_top_p,
                                 reference_temperature), 0, None, True,
                reference_temperature),
            "mass-global": (
                calibrated_metrics(rows, reference, test, global_model,
                                   candidate_temperature, reference_temperature),
                1, global_model, True, candidate_temperature),
            "mass-conditioned-forced": (
                calibrated_metrics(rows, reference, test, conditioned_model,
                                   candidate_temperature, reference_temperature),
                groups, conditioned_model, True, candidate_temperature),
            "mass-conditioned-selected": (
                calibrated_metrics(rows, reference, test,
                                   conditioned_model if enabled else global_model,
                                   candidate_temperature, reference_temperature), groups,
                conditioned_model if enabled else global_model, enabled,
                candidate_temperature),
        }
        for method in ("tuned-temperature", "tuned-top-p", "tuned-temperature-top-p"):
            if fold in settings and method in settings[fold]:
                temperature, top_p = settings[fold][method]
                methods[method] = (
                    baseline_metrics(rows, reference, orders, test, temperature, top_p,
                                     reference_temperature),
                    0, None, True, temperature)
        for method, (metrics, selected_groups, model, selected_enabled,
                     applied_temperature) in methods.items():
            candidate_top_ps = ("+".join(f"{value:.6f}" for value in model.candidate_top_ps)
                                 if model is not None else "")
            entropy_thresholds = ("+".join(f"{value:.6f}" for value in model.entropy_thresholds)
                                  if model is not None else "")
            output.append({
                "quant": name, "fold": fold,
                "test_chunks": "+".join(map(str, test_chunks)),
                "inner_validation_chunks": "+".join(map(str, inner_validation_chunks)),
                "train_positions": int(train.sum()), "test_positions": int(test.sum()),
                "method": method, "reference_temperature": reference_temperature,
                "reference_top_p": reference_top_p,
                "candidate_temperature": applied_temperature,
                "selected_groups": selected_groups, "selected_enabled": selected_enabled,
                "entropy_thresholds": entropy_thresholds,
                "candidate_top_ps": candidate_top_ps, **metrics,
            })
        print(f"finished {name} fold {fold}: groups={groups}, enabled={enabled}", flush=True)
    return output


def write_report(path: Path, rows: list[dict], reference_label: str) -> None:
    baseline = np.mean([row["sampler_js"] for row in rows
                        if row["method"] == "same-settings"])
    preferred_order = [
        "same-settings", "tuned-temperature", "tuned-top-p",
        "tuned-temperature-top-p", "mass-global", "mass-conditioned-forced",
        "mass-conditioned-selected",
    ]
    methods = [method for method in preferred_order
               if any(row["method"] == method for row in rows)]
    lines = [
        "# Cumulative-mass calibrated nucleus sampler", "",
        f"The target is {reference_label} at T=0.8/top-p=0.95. Calibration measures how much",
        "reference probability lies in prefixes sorted by the quant candidate, then inverts",
        "that monotonic curve to choose a candidate top-p. The conditioned model fits separate",
        "curves in quant-entropy bins. Bin count is selected on inner held-out chunks and may",
        "fall back to the global curve. Scalar baselines are the independently cross-validated",
        "settings from the established sampler-grid experiment.", "",
        "| Method | JS | Recovered | TV | Support Jaccard | Ref mass retained | Mean candidate top-p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        js = np.mean([row["sampler_js"] for row in selected])
        lines.append(
            f"| {method} | {js:.7f} | {1.0 - js / baseline:.2%} | "
            f"{np.mean([row['sampler_tv'] for row in selected]):.7f} | "
            f"{np.mean([row['sampler_support_jaccard'] for row in selected]):.1%} | "
            f"{np.mean([row['reference_mass_in_candidate_support'] for row in selected]):.2%} | "
            f"{np.mean([row['mean_candidate_top_p'] for row in selected]):.4f} |")
    lines += ["", "## Inner-selected conditioning", "",
              "| Fold | Entropy groups | Enabled | Candidate top-p values |",
              "|---:|---:|:---:|---|" ]
    selected_rows = [row for row in rows if row["method"] == "mass-conditioned-selected"]
    for row in selected_rows:
        lines.append(
            f"| {row['fold']} | {row['selected_groups']} | "
            f"{'yes' if row['selected_enabled'] else 'no'} | {row['candidate_top_ps']} |")
    lines += ["", "Fold-level values are in `mass_calibration.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--reference-temperature", type=float, default=0.8)
    parser.add_argument("--reference-top-p", type=float, default=0.95)
    parser.add_argument("--candidate-temperature", type=float, default=0.8)
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--mass-points", type=int, default=240)
    parser.add_argument("--sampler-settings-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(
        args.reference, args.candidate, args.candidate_name, args.folds,
        args.reference_temperature, args.reference_top_p, args.candidate_temperature,
        args.groups, args.mass_points, args.sampler_settings_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "mass_calibration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir / "MASS_CALIBRATION.md", rows, args.reference_label)
    print(f"wrote {args.output_dir / 'MASS_CALIBRATION.md'}")


if __name__ == "__main__":
    main()
