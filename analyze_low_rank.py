#!/usr/bin/env python3
"""Nested held-out test of a head-restricted low-rank logit denoiser."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_sparse import (PairedRow, fit_temperature, fit_token_bias, load_sparse,
                            softmax)


@dataclass
class BasisModel:
    token_to_column: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray


@dataclass
class RidgeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray


def head_positions(row: PairedRow, count: int) -> np.ndarray:
    count = min(count, len(row.token_ids))
    reference = np.argpartition(row.reference, -count)[-count:]
    candidate = np.argpartition(row.candidate, -count)[-count:]
    positions = np.union1d(reference, candidate)
    return positions[row.exact[positions]]


def base_and_residual(row: PairedRow, scale: float, bias: np.ndarray,
                      temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = softmax(row.reference / temperature)
    base = scale * row.candidate + bias[row.token_ids]
    residual = row.reference - base
    residual -= float(np.dot(p, residual))
    return p, base, residual


def fit_basis(rows: list[PairedRow], selected: np.ndarray, n_vocab: int, scale: float,
              bias: np.ndarray, temperature: float, head: int, components: int,
              seed: int) -> BasisModel:
    indices = np.flatnonzero(selected)
    heads = [head_positions(rows[index], head) for index in indices]
    tokens = np.unique(np.concatenate([
        rows[index].token_ids[positions] for index, positions in zip(indices, heads)
    ]))
    token_to_column = np.full(n_vocab, -1, dtype=np.int32)
    token_to_column[tokens] = np.arange(len(tokens), dtype=np.int32)
    matrix = np.zeros((len(indices), len(tokens)), dtype=np.float32)
    for output_row, (index, positions) in enumerate(zip(indices, heads)):
        _, _, residual = base_and_residual(rows[index], scale, bias, temperature)
        columns = token_to_column[rows[index].token_ids[positions]]
        matrix[output_row, columns] = residual[positions]
    matrix -= matrix.mean(axis=0, keepdims=True)

    rank = min(components, matrix.shape[0] - 1, matrix.shape[1])
    if rank <= 0:
        return BasisModel(token_to_column, np.empty((0, len(tokens)), dtype=np.float32),
                          np.empty(0, dtype=np.float32))
    sketch_rank = min(rank + 12, matrix.shape[0] - 1, matrix.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], sketch_rank), dtype=np.float32)
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    small = q.T @ matrix
    _, singular_values, vt = np.linalg.svd(small, full_matrices=False)
    return BasisModel(token_to_column, vt[:rank].astype(np.float32),
                      singular_values[:rank].astype(np.float32))


def row_basis(row: PairedRow, model: BasisModel, components: int) -> np.ndarray:
    result = np.zeros((len(row.token_ids), components), dtype=np.float32)
    columns = model.token_to_column[row.token_ids]
    valid = columns >= 0
    if np.any(valid) and components:
        result[valid] = model.basis[:components, columns[valid]].T
    return result


def oracle_coefficients(row: PairedRow, model: BasisModel, components: int, scale: float,
                        bias: np.ndarray, temperature: float) -> np.ndarray:
    if components == 0:
        return np.empty(0, dtype=np.float32)
    p, base, _ = base_and_residual(row, scale, bias, temperature)
    directions = row_basis(row, model, components).astype(np.float64)
    target = np.dot(p, directions)
    coefficients = np.zeros(components, dtype=np.float64)

    def objective(values: np.ndarray) -> float:
        logits = (base + directions @ values) / temperature
        maximum = float(logits.max())
        return maximum + np.log(np.exp(logits - maximum).sum()) - float(np.dot(p, logits))

    value = objective(coefficients)
    for _ in range(8):
        logits = (base + directions @ coefficients) / temperature
        maximum = float(logits.max())
        q = np.exp(logits - maximum)
        q /= q.sum()
        mean = np.dot(q, directions)
        gradient = (mean - target) / temperature
        centered = directions - mean[None, :]
        hessian = centered.T @ (q[:, None] * centered) / (temperature * temperature)
        regularizer = max(float(np.trace(hessian)) / max(components, 1), 1e-8) * 1e-5
        step = np.linalg.solve(hessian + regularizer * np.eye(components), gradient)
        if np.linalg.norm(step) < 1e-5:
            break
        step_scale = 1.0
        accepted = False
        while step_scale >= 1 / 128:
            candidate = coefficients - step_scale * step
            candidate_value = objective(candidate)
            if candidate_value <= value:
                coefficients, value = candidate, candidate_value
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
    return coefficients.astype(np.float32)


def quant_features(row: PairedRow, model: BasisModel, components: int,
                   temperature: float) -> np.ndarray:
    q = softmax(row.candidate / temperature)
    order = np.argsort(-row.candidate)
    sorted_logits = row.candidate[order]
    sorted_probabilities = q[order]
    mean = float(np.dot(q, row.candidate))
    variance = float(np.dot(q, (row.candidate - mean) ** 2))
    basic = np.array([
        -float(np.dot(q, np.log(q))),
        np.sqrt(max(variance, 0.0)),
        float(sorted_probabilities[0]),
        float(sorted_probabilities[:5].sum()),
        float(sorted_probabilities[:20].sum()),
        float(sorted_logits[0] - sorted_logits[1]),
        float(sorted_logits[0] - sorted_logits[min(4, len(order) - 1)]),
        float(sorted_logits[0] - sorted_logits[min(19, len(order) - 1)]),
    ], dtype=np.float32)
    if components == 0:
        return basic
    directions = row_basis(row, model, components)
    expected_direction = np.dot(q, directions)
    logit_covariance = np.dot(q * (row.candidate - mean), directions)
    top_direction = directions[order[0]]
    return np.concatenate([basic, expected_direction, logit_covariance, top_direction])


def ridge_training_data(rows: list[PairedRow], selected: np.ndarray, model: BasisModel,
                        components: int, scale: float, bias: np.ndarray,
                        temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(selected)
    features = np.stack([
        quant_features(rows[index], model, components, temperature) for index in indices
    ]).astype(np.float64)
    targets = np.stack([
        oracle_coefficients(rows[index], model, components, scale, bias, temperature)
        for index in indices
    ]).astype(np.float64)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(indices)), standardized])
    return (feature_mean.astype(np.float32), feature_scale.astype(np.float32), design,
            targets)


def solve_ridge(feature_mean: np.ndarray, feature_scale: np.ndarray, design: np.ndarray,
                targets: np.ndarray, alpha: float) -> RidgeModel:
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return RidgeModel(feature_mean, feature_scale, coefficients.astype(np.float32))


def fit_ridge(rows: list[PairedRow], selected: np.ndarray, model: BasisModel, components: int,
              scale: float, bias: np.ndarray, temperature: float, alpha: float) -> RidgeModel:
    return solve_ridge(*ridge_training_data(
        rows, selected, model, components, scale, bias, temperature), alpha)


def predict_coefficients(row: PairedRow, basis: BasisModel, ridge: RidgeModel,
                         components: int, temperature: float) -> np.ndarray:
    features = quant_features(row, basis, components, temperature)
    standardized = (features - ridge.feature_mean) / ridge.feature_scale
    return np.concatenate([[1.0], standardized]) @ ridge.coefficients


def evaluate(rows: list[PairedRow], selected: np.ndarray, temperature: float, scale: float,
             bias: np.ndarray, basis: BasisModel | None = None, components: int = 0,
             ridge: RidgeModel | None = None, oracle: bool = False) -> dict[str, float]:
    kl, js, tv, top1, top10 = [], [], [], [], []
    for index in np.flatnonzero(selected):
        row = rows[index]
        reference_scaled = row.reference.astype(np.float64) / temperature
        log_p = reference_scaled - (
            reference_scaled.max() + np.log(np.exp(reference_scaled - reference_scaled.max()).sum()))
        p = np.exp(log_p)
        corrected = scale * row.candidate + bias[row.token_ids]
        if basis is not None and components:
            if oracle:
                coefficients = oracle_coefficients(
                    row, basis, components, scale, bias, temperature)
            else:
                assert ridge is not None
                coefficients = predict_coefficients(
                    row, basis, ridge, components, temperature)
            corrected = corrected + row_basis(row, basis, components) @ coefficients
        candidate_scaled = corrected.astype(np.float64) / temperature
        log_q = candidate_scaled - (
            candidate_scaled.max() + np.log(np.exp(candidate_scaled - candidate_scaled.max()).sum()))
        q = np.exp(log_q)
        kl.append(float(np.dot(p, log_p - log_q)))
        midpoint = 0.5 * (p + q)
        p_keep, q_keep = p > 0, q > 0
        js.append(float(0.5 * np.dot(p[p_keep], log_p[p_keep] - np.log(midpoint[p_keep]))
                        + 0.5 * np.dot(q[q_keep], log_q[q_keep] - np.log(midpoint[q_keep]))))
        tv.append(float(0.5 * np.abs(p - q).sum()))
        ref_top = np.argpartition(row.reference, -10)[-10:]
        corrected_top = np.argpartition(corrected, -10)[-10:]
        top1.append(int(np.argmax(row.reference) == np.argmax(corrected)))
        top10.append(np.intersect1d(ref_top, corrected_top, assume_unique=True).size / 10.0)
    return {
        "kl": float(np.mean(kl)), "js": float(np.mean(js)), "tv": float(np.mean(tv)),
        "top1_agreement": float(np.mean(top1)), "top10_overlap": float(np.mean(top10)),
    }


def select_hyperparameters(rows: list[PairedRow], inner_train: np.ndarray,
                           inner_validation: np.ndarray, n_vocab: int, temperature: float,
                           head: int, components_grid: list[int], alpha_grid: list[float],
                           seed: int) -> tuple[int, float]:
    scale = fit_temperature(rows, inner_train, temperature)
    bias = fit_token_bias(rows, inner_train, n_vocab, scale, temperature)
    max_components = max(components_grid)
    basis = fit_basis(rows, inner_train, n_vocab, scale, bias, temperature, head,
                      max_components, seed)
    baseline = evaluate(rows, inner_validation, temperature, scale, bias)["kl"]
    best = (baseline, 0, alpha_grid[0])
    for components in components_grid:
        if components == 0 or components > len(basis.basis):
            continue
        training_data = ridge_training_data(
            rows, inner_train, basis, components, scale, bias, temperature)
        for alpha in alpha_grid:
            ridge = solve_ridge(*training_data, alpha)
            value = evaluate(rows, inner_validation, temperature, scale, bias, basis,
                             components, ridge)["kl"]
            candidate = (value, components, alpha)
            if candidate < best:
                best = candidate
    return int(best[1]), float(best[2])


def analyze(reference_path: Path, candidate_path: Path, name: str, temperature: float,
            folds: int, head: int, components_grid: list[int],
            alpha_grid: list[float]) -> tuple[list[dict], list[tuple[int, float]]]:
    rows, chunks, n_vocab, n_chunks = load_sparse(reference_path, candidate_path)
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    output, selections = [], []
    max_components = max(components_grid)
    zero_bias = np.zeros(n_vocab, dtype=np.float32)
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
        selected_components, selected_alpha = select_hyperparameters(
            rows, inner_train, inner_validation, n_vocab, temperature, head,
            components_grid, alpha_grid, seed=1000 + fold)
        selections.append((selected_components, selected_alpha))

        scale = fit_temperature(rows, train, temperature)
        bias = fit_token_bias(rows, train, n_vocab, scale, temperature)
        basis = fit_basis(rows, train, n_vocab, scale, bias, temperature, head,
                          max_components, seed=2000 + fold)
        ridge = None
        if selected_components:
            ridge = fit_ridge(rows, train, basis, selected_components, scale, bias,
                              temperature, selected_alpha)
        methods = {
            "identity": evaluate(rows, test, temperature, 1.0, zero_bias),
            "target-temperature": evaluate(rows, test, temperature, scale, zero_bias),
            "target-temperature+token": evaluate(rows, test, temperature, scale, bias),
            "low-rank-predicted": evaluate(rows, test, temperature, scale, bias, basis,
                                           selected_components, ridge),
            f"low-rank-oracle-{max_components}": evaluate(
                rows, test, temperature, scale, bias, basis, len(basis.basis), oracle=True),
        }
        for method, metrics in methods.items():
            output.append({
                "quant": name, "fold": fold,
                "test_chunks": "+".join(map(str, test_chunks)),
                "inner_validation_chunks": "+".join(map(str, inner_validation_chunks)),
                "train_positions": int(train.sum()), "test_positions": int(test.sum()),
                "temperature": temperature, "head": head, "method": method,
                "selected_components": selected_components,
                "selected_alpha": selected_alpha, "scale": scale, **metrics,
            })
        print(f"finished {name} fold {fold}: k={selected_components}, alpha={selected_alpha:g}",
              flush=True)
    return output, selections


def write_report(path: Path, rows: list[dict], reference_label: str, head: int) -> None:
    identity = np.mean([row["kl"] for row in rows if row["method"] == "identity"])
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Head-restricted low-rank denoiser",
        "",
        f"The target is {reference_label}. Residual directions are learned only on the union of",
        f"the reference and candidate top-{head} heads. Each outer fold holds out complete chunks;",
        "the number of directions and ridge penalty are selected on an inner chunk split.",
        "The predicted method receives quant-only distribution features. The oracle uses the",
        "held-out reference to choose amplitudes and is an unattainable diagnostic upper bound.",
        "",
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
    lines += ["", "## Inner-selected capacity", "",
              "| Fold | Directions | Ridge alpha |", "|---:|---:|---:|"]
    predicted = [row for row in rows if row["method"] == "low-rank-predicted"]
    for row in predicted:
        lines.append(f"| {row['fold']} | {row['selected_components']} | "
                     f"{row['selected_alpha']:g} |")
    lines += ["", "Fold-level values are in `low_rank.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--components", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--ridge-alpha", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, _ = analyze(args.reference, args.candidate, args.candidate_name,
                      args.temperature, args.folds, args.head, args.components,
                      args.ridge_alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "low_rank.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir / "LOW_RANK.md", rows, args.reference_label, args.head)
    print(f"wrote {args.output_dir / 'LOW_RANK.md'}")


if __name__ == "__main__":
    main()
