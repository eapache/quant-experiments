#!/usr/bin/env python3
"""Nested held-out test of a tiny nonlinear predictor for low-rank logit residuals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_low_rank import (BasisModel, evaluate, fit_basis, oracle_coefficients,
                              quant_features, row_basis)
from analyze_sparse import PairedRow, fit_temperature, fit_token_bias, load_sparse, softmax


@dataclass
class MLPModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def top_order(row: PairedRow, count: int) -> np.ndarray:
    count = min(count, len(row.token_ids))
    if count == len(row.token_ids):
        return np.argsort(-row.candidate)
    positions = np.argpartition(row.candidate, -count)[-count:]
    return positions[np.argsort(-row.candidate[positions])]


def rich_features(row: PairedRow, basis: BasisModel, components: int,
                  temperature: float, shape_head: int) -> np.ndarray:
    features = quant_features(row, basis, components, temperature)
    q = softmax(row.candidate / temperature)
    order = top_order(row, shape_head)
    gaps = np.full(shape_head, 16.0, dtype=np.float32)
    probabilities = np.zeros(shape_head, dtype=np.float32)
    gaps[:len(order)] = np.clip(row.candidate[order[0]] - row.candidate[order], 0.0, 16.0)
    probabilities[:len(order)] = q[order]
    return np.concatenate([features, gaps, probabilities])


def training_data(rows: list[PairedRow], selected: np.ndarray, basis: BasisModel,
                  components: int, scale: float, bias: np.ndarray, temperature: float,
                  shape_head: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(selected)
    x = np.stack([
        rich_features(rows[index], basis, components, temperature, shape_head)
        for index in indices
    ]).astype(np.float64)
    y = np.stack([
        oracle_coefficients(rows[index], basis, components, scale, bias, temperature)
        for index in indices
    ]).astype(np.float64)
    return x, y


def initialize_model(x: np.ndarray, y: np.ndarray, hidden: int,
                     seed: int) -> tuple[MLPModel, np.ndarray, np.ndarray]:
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    target_mean = y.mean(axis=0)
    target_scale = y.std(axis=0)
    target_scale[target_scale < 1e-6] = 1.0
    xs = (x - feature_mean) / feature_scale
    ys = np.clip((y - target_mean) / target_scale, -5.0, 5.0)
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((x.shape[1], hidden)) / np.sqrt(x.shape[1])
    w2 = rng.standard_normal((hidden, y.shape[1])) / np.sqrt(hidden)
    model = MLPModel(feature_mean, feature_scale, target_mean, target_scale,
                     w1, np.zeros(hidden), w2, np.zeros(y.shape[1]))
    return model, xs, ys


def copy_model(model: MLPModel) -> MLPModel:
    return MLPModel(*(value.copy() for value in (
        model.feature_mean, model.feature_scale, model.target_mean, model.target_scale,
        model.w1, model.b1, model.w2, model.b2)))


def train_mlp(x: np.ndarray, y: np.ndarray, hidden: int, weight_decay: float,
              checkpoints: list[int], seed: int) -> dict[int, MLPModel]:
    model, xs, ys = initialize_model(x, y, hidden, seed)
    parameters = [model.w1, model.b1, model.w2, model.b2]
    first = [np.zeros_like(parameter) for parameter in parameters]
    second = [np.zeros_like(parameter) for parameter in parameters]
    snapshots: dict[int, MLPModel] = {}
    learning_rate = 0.01
    for step in range(1, max(checkpoints) + 1):
        hidden_values = np.tanh(xs @ model.w1 + model.b1)
        prediction = hidden_values @ model.w2 + model.b2
        output_gradient = (prediction - ys) / len(xs)
        gradients = [
            xs.T @ ((output_gradient @ model.w2.T) * (1.0 - hidden_values ** 2))
            + weight_decay * model.w1,
            np.sum((output_gradient @ model.w2.T) * (1.0 - hidden_values ** 2), axis=0),
            hidden_values.T @ output_gradient + weight_decay * model.w2,
            output_gradient.sum(axis=0),
        ]
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first[index] = 0.9 * first[index] + 0.1 * gradient
            second[index] = 0.999 * second[index] + 0.001 * gradient ** 2
            first_hat = first[index] / (1.0 - 0.9 ** step)
            second_hat = second[index] / (1.0 - 0.999 ** step)
            parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        if step in checkpoints:
            snapshots[step] = copy_model(model)
    return snapshots


def predict(row: PairedRow, basis: BasisModel, model: MLPModel, components: int,
            temperature: float, shape_head: int) -> np.ndarray:
    features = rich_features(row, basis, components, temperature, shape_head)
    standardized = (features - model.feature_mean) / model.feature_scale
    hidden = np.tanh(standardized @ model.w1 + model.b1)
    output = hidden @ model.w2 + model.b2
    output = np.clip(output, -5.0, 5.0)
    return (model.target_mean + model.target_scale * output).astype(np.float32)


def evaluate_neural(rows: list[PairedRow], selected: np.ndarray, temperature: float,
                    scale: float, bias: np.ndarray, basis: BasisModel, model: MLPModel,
                    components: int, shape_head: int) -> dict[str, float]:
    values = {key: [] for key in ("kl", "js", "tv", "top1_agreement", "top10_overlap")}
    for index in np.flatnonzero(selected):
        row = rows[index]
        reference_scaled = row.reference.astype(np.float64) / temperature
        log_p = reference_scaled - (
            reference_scaled.max() + np.log(np.exp(
                reference_scaled - reference_scaled.max()).sum()))
        p = np.exp(log_p)
        corrected = scale * row.candidate + bias[row.token_ids]
        coefficients = predict(row, basis, model, components, temperature, shape_head)
        corrected = corrected + row_basis(row, basis, components) @ coefficients
        candidate_scaled = corrected.astype(np.float64) / temperature
        log_q = candidate_scaled - (
            candidate_scaled.max() + np.log(np.exp(
                candidate_scaled - candidate_scaled.max()).sum()))
        q = np.exp(log_q)
        values["kl"].append(float(np.dot(p, log_p - log_q)))
        midpoint = 0.5 * (p + q)
        values["js"].append(float(
            0.5 * np.dot(p, log_p - np.log(midpoint))
            + 0.5 * np.dot(q, log_q - np.log(midpoint))))
        values["tv"].append(float(0.5 * np.abs(p - q).sum()))
        reference_top = np.argpartition(row.reference, -10)[-10:]
        corrected_top = np.argpartition(corrected, -10)[-10:]
        values["top1_agreement"].append(
            float(np.argmax(row.reference) == np.argmax(corrected)))
        values["top10_overlap"].append(
            float(np.intersect1d(reference_top, corrected_top, assume_unique=True).size / 10.0))
    return {key: float(np.mean(value)) for key, value in values.items()}


def select_config(rows: list[PairedRow], inner_train: np.ndarray,
                  inner_validation: np.ndarray, n_vocab: int, temperature: float,
                  head: int, components: int, shape_head: int, hidden_grid: list[int],
                  decay_grid: list[float], checkpoints: list[int], seed: int
                  ) -> tuple[int, float, int, bool]:
    scale = fit_temperature(rows, inner_train, temperature)
    bias = fit_token_bias(rows, inner_train, n_vocab, scale, temperature)
    basis = fit_basis(rows, inner_train, n_vocab, scale, bias, temperature,
                      head, components, seed)
    x, y = training_data(rows, inner_train, basis, components, scale, bias,
                         temperature, shape_head)
    baseline = evaluate(rows, inner_validation, temperature, scale, bias)["kl"]
    best = (float("inf"), hidden_grid[0], decay_grid[0], checkpoints[0])
    for hidden in hidden_grid:
        for decay_index, decay in enumerate(decay_grid):
            snapshots = train_mlp(
                x, y, hidden, decay, checkpoints,
                seed + 100 * hidden + decay_index)
            for step, model in snapshots.items():
                kl = evaluate_neural(
                    rows, inner_validation, temperature, scale, bias, basis, model,
                    components, shape_head)["kl"]
                best = min(best, (kl, hidden, decay, step))
    return int(best[1]), float(best[2]), int(best[3]), bool(best[0] < baseline)


def analyze(reference_path: Path, candidate_path: Path, name: str, temperature: float,
            folds: int, head: int, components: int, shape_head: int,
            hidden_grid: list[int], decay_grid: list[float], checkpoints: list[int]
            ) -> list[dict]:
    rows, chunks, n_vocab, n_chunks = load_sparse(reference_path, candidate_path)
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    zero_bias = np.zeros(n_vocab, dtype=np.float32)
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
        hidden, decay, steps, enabled = select_config(
            rows, inner_train, inner_validation, n_vocab, temperature, head,
            components, shape_head, hidden_grid, decay_grid, checkpoints, 1000 + fold)

        scale = fit_temperature(rows, train, temperature)
        bias = fit_token_bias(rows, train, n_vocab, scale, temperature)
        basis = fit_basis(rows, train, n_vocab, scale, bias, temperature,
                          head, components, 2000 + fold)
        x, y = training_data(rows, train, basis, components, scale, bias,
                             temperature, shape_head)
        model = train_mlp(x, y, hidden, decay, [steps], 3000 + fold)[steps]
        baseline = evaluate(rows, test, temperature, scale, bias)
        neural = evaluate_neural(rows, test, temperature, scale, bias, basis, model,
                                 components, shape_head)
        methods = {
            "identity": evaluate(rows, test, temperature, 1.0, zero_bias),
            "target-temperature": evaluate(rows, test, temperature, scale, zero_bias),
            "target-temperature+token": baseline,
            "neural-low-rank-forced": neural,
            "neural-low-rank-selected": neural if enabled else baseline,
            f"low-rank-oracle-{components}": evaluate(
                rows, test, temperature, scale, bias, basis, components, oracle=True),
        }
        for method, metrics in methods.items():
            output.append({
                "quant": name, "fold": fold,
                "test_chunks": "+".join(map(str, test_chunks)),
                "inner_validation_chunks": "+".join(map(str, inner_validation_chunks)),
                "train_positions": int(train.sum()), "test_positions": int(test.sum()),
                "temperature": temperature, "head": head, "components": components,
                "shape_head": shape_head, "method": method,
                "selected_hidden": hidden, "selected_weight_decay": decay,
                "selected_steps": steps, "selected_enabled": enabled,
                "parameter_count": ((len(model.feature_mean) + 1) * hidden
                                    + (hidden + 1) * components),
                "scale": scale, **metrics,
            })
        print(
            f"finished {name} fold {fold}: hidden={hidden}, decay={decay:g}, "
            f"steps={steps}, enabled={enabled}", flush=True)
    return output


def write_report(path: Path, rows: list[dict], reference_label: str) -> None:
    identity = np.mean([row["kl"] for row in rows if row["method"] == "identity"])
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Tiny nonlinear low-rank predictor",
        "",
        f"The target is {reference_label}. A one-hidden-layer tanh network reads quant-only",
        "top-logit gaps/probabilities plus projections of the candidate distribution onto the",
        "learned residual basis, then predicts 16 context-specific correction amplitudes.",
        "Hidden width, weight decay, and training duration are selected on inner held-out",
        "chunks. The selected variant may reject the network in favor of static token bias.",
        "The forced variant reports the best non-null inner configuration on untouched outer",
        "chunks. The oracle remains unattainable and uses each held-out BF16 row.",
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
    lines += [
        "", "## Inner-selected configurations", "",
        "| Fold | Hidden | Parameters | Weight decay | Steps | Enabled |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    selected_rows = [row for row in rows if row["method"] == "neural-low-rank-selected"]
    for row in selected_rows:
        lines.append(
            f"| {row['fold']} | {row['selected_hidden']} | {row['parameter_count']} | "
            f"{row['selected_weight_decay']:g} | {row['selected_steps']} | "
            f"{'yes' if row['selected_enabled'] else 'no'} |")
    lines += ["", "Fold-level values are in `neural_low_rank.csv`.", ""]
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
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--shape-head", type=int, default=64)
    parser.add_argument("--hidden", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(
        args.reference, args.candidate, args.candidate_name, args.temperature, args.folds,
        args.head, args.components, args.shape_head, args.hidden, args.weight_decay,
        args.checkpoints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "neural_low_rank.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir / "NEURAL_LOW_RANK.md", rows, args.reference_label)
    print(f"wrote {args.output_dir / 'NEURAL_LOW_RANK.md'}")


if __name__ == "__main__":
    main()
