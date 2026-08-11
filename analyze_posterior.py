#!/usr/bin/env python3
"""Nested held-out test of a conditional empirical posterior over Q2 head errors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_sparse import PairedRow, fit_temperature, fit_token_bias, load_sparse, softmax


@dataclass
class ResidualBank:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    features: np.ndarray
    residuals: np.ndarray


def top_order(row: PairedRow, count: int) -> np.ndarray:
    count = min(count, len(row.token_ids))
    if count == len(row.token_ids):
        return np.argsort(-row.candidate)
    positions = np.argpartition(row.candidate, -count)[-count:]
    return positions[np.argsort(-row.candidate[positions])]


def quant_features(row: PairedRow, temperature: float, gap_features: int = 16) -> np.ndarray:
    q = softmax(row.candidate / temperature)
    order = top_order(row, max(20, gap_features + 1))
    logits = row.candidate[order]
    probabilities = q[order]
    mean = float(np.dot(q, row.candidate))
    variance = float(np.dot(q, (row.candidate - mean) ** 2))
    basics = np.array([
        -float(np.dot(q, np.log(q))),
        np.sqrt(max(variance, 0.0)),
        float(probabilities[0]),
        float(probabilities[:5].sum()),
        float(probabilities[:20].sum()),
    ], dtype=np.float32)
    gaps = np.full(gap_features, 16.0, dtype=np.float32)
    available = min(gap_features, len(logits) - 1)
    if available:
        gaps[:available] = np.clip(logits[0] - logits[1:available + 1], 0.0, 16.0)
    return np.concatenate([basics, gaps])


def centered_residual(row: PairedRow, scale: float, bias: np.ndarray,
                      temperature: float) -> np.ndarray:
    p = softmax(row.reference / temperature)
    residual = row.reference - (scale * row.candidate + bias[row.token_ids])
    residual -= float(np.dot(p, residual))
    return residual


def fit_bank(rows: list[PairedRow], selected: np.ndarray, scale: float, bias: np.ndarray,
             temperature: float, max_head: int) -> ResidualBank:
    indices = np.flatnonzero(selected)
    features = np.stack([quant_features(rows[index], temperature) for index in indices])
    residuals = np.full((len(indices), max_head), np.nan, dtype=np.float32)
    for output_row, index in enumerate(indices):
        row = rows[index]
        order = top_order(row, max_head)
        residual = centered_residual(row, scale, bias, temperature)
        valid = row.exact[order]
        residuals[output_row, :len(order)][valid] = residual[order[valid]]
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    return ResidualBank(
        feature_mean.astype(np.float32), feature_scale.astype(np.float32),
        ((features - feature_mean) / feature_scale).astype(np.float32), residuals)


def nearest_residuals(row: PairedRow, bank: ResidualBank, neighbors: int,
                      temperature: float, head: int) -> np.ndarray:
    feature = (quant_features(row, temperature) - bank.feature_mean) / bank.feature_scale
    distances = np.sum((bank.features - feature) ** 2, axis=1)
    neighbors = min(neighbors, len(distances))
    if neighbors == len(distances):
        nearest = np.arange(len(distances))
    else:
        nearest = np.argpartition(distances, neighbors - 1)[:neighbors]
    return bank.residuals[nearest, :head].astype(np.float64)


def predictive_distribution(row: PairedRow, scale: float, bias: np.ndarray,
                            temperature: float, bank: ResidualBank, head: int,
                            neighbors: int, posterior: bool) -> np.ndarray:
    base_logits = (scale * row.candidate + bias[row.token_ids]).astype(np.float64)
    base = softmax(base_logits / temperature).astype(np.float64)
    order = top_order(row, head)
    samples = nearest_residuals(row, bank, neighbors, temperature, len(order))
    valid = np.isfinite(samples)
    counts = valid.sum(axis=0)
    totals = np.nansum(samples, axis=0)
    mean = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    mean = np.clip(mean, -4.0, 4.0)
    if not posterior:
        result = base.copy()
        multipliers = np.exp(mean / temperature)
        normalizer = 1.0 + float(np.dot(base[order], multipliers - 1.0))
        result /= normalizer
        result[order] *= multipliers
        return result

    samples = np.where(valid, samples, mean[None, :])
    samples = np.clip(samples, -4.0, 4.0)
    multipliers = np.exp(samples / temperature)
    normalizers = 1.0 + np.sum(base[order][None, :] * (multipliers - 1.0), axis=1)
    inverse_mean = float(np.mean(1.0 / normalizers))
    result = base * inverse_mean
    result[order] = base[order] * np.mean(multipliers / normalizers[:, None], axis=0)
    result /= result.sum()
    return result


def distribution_metrics(row: PairedRow, q: np.ndarray,
                         temperature: float) -> tuple[float, float, float, float, float]:
    p = softmax(row.reference / temperature).astype(np.float64)
    log_p = np.log(p)
    log_q = np.log(q)
    kl = float(np.dot(p, log_p - log_q))
    midpoint = 0.5 * (p + q)
    js = float(0.5 * np.dot(p, log_p - np.log(midpoint))
               + 0.5 * np.dot(q, log_q - np.log(midpoint)))
    tv = float(0.5 * np.abs(p - q).sum())
    ref_top = np.argpartition(row.reference, -min(10, len(row.reference)))[-10:]
    candidate_top = np.argpartition(q, -min(10, len(q)))[-10:]
    top1 = float(np.argmax(row.reference) == np.argmax(q))
    top10 = float(np.intersect1d(ref_top, candidate_top, assume_unique=True).size / 10.0)
    return kl, js, tv, top1, top10


def evaluate_base(rows: list[PairedRow], selected: np.ndarray, temperature: float,
                  scale: float, bias: np.ndarray) -> dict[str, float]:
    values = []
    for index in np.flatnonzero(selected):
        row = rows[index]
        q = softmax((scale * row.candidate + bias[row.token_ids]) / temperature)
        values.append(distribution_metrics(row, q, temperature))
    array = np.asarray(values)
    return dict(zip(("kl", "js", "tv", "top1_agreement", "top10_overlap"), array.mean(0)))


def evaluate_predictive(rows: list[PairedRow], selected: np.ndarray, temperature: float,
                        scale: float, bias: np.ndarray, bank: ResidualBank, head: int,
                        neighbors: int, posterior: bool) -> dict[str, float]:
    values = []
    for index in np.flatnonzero(selected):
        row = rows[index]
        q = predictive_distribution(
            row, scale, bias, temperature, bank, head, neighbors, posterior)
        values.append(distribution_metrics(row, q, temperature))
    array = np.asarray(values)
    return dict(zip(("kl", "js", "tv", "top1_agreement", "top10_overlap"), array.mean(0)))


def select_configs(rows: list[PairedRow], inner_train: np.ndarray,
                   inner_validation: np.ndarray, n_vocab: int, temperature: float,
                   heads: list[int], neighbors_grid: list[int]) -> dict[bool, tuple[int, int, bool]]:
    scale = fit_temperature(rows, inner_train, temperature)
    bias = fit_token_bias(rows, inner_train, n_vocab, scale, temperature)
    bank = fit_bank(rows, inner_train, scale, bias, temperature, max(heads))
    baseline = evaluate_base(rows, inner_validation, temperature, scale, bias)["kl"]
    best: dict[bool, tuple[float, int, int]] = {
        False: (float("inf"), heads[0], neighbors_grid[0]),
        True: (float("inf"), heads[0], neighbors_grid[0]),
    }
    for head in heads:
        for neighbors in neighbors_grid:
            for posterior in (False, True):
                metrics = evaluate_predictive(
                    rows, inner_validation, temperature, scale, bias, bank,
                    head, neighbors, posterior)
                candidate = (metrics["kl"], head, neighbors)
                if candidate < best[posterior]:
                    best[posterior] = candidate
    return {
        posterior: (value[1], value[2], value[0] < baseline)
        for posterior, value in best.items()
    }


def analyze(reference_path: Path, candidate_path: Path, name: str, temperature: float,
            folds: int, heads: list[int], neighbors_grid: list[int]) -> list[dict]:
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
        configs = select_configs(
            rows, inner_train, inner_validation, n_vocab, temperature, heads, neighbors_grid)

        scale = fit_temperature(rows, train, temperature)
        bias = fit_token_bias(rows, train, n_vocab, scale, temperature)
        bank = fit_bank(rows, train, scale, bias, temperature, max(heads))
        methods = {
            "identity": (evaluate_base(rows, test, temperature, 1.0, zero_bias), 0, 0),
            "target-temperature": (
                evaluate_base(rows, test, temperature, scale, zero_bias), 0, 0),
            "target-temperature+token": (
                evaluate_base(rows, test, temperature, scale, bias), 0, 0),
        }
        for posterior, method in ((False, "conditional-mean-forced"),
                                  (True, "posterior-predictive-forced")):
            head, neighbors, enabled = configs[posterior]
            predictive = (
                evaluate_predictive(rows, test, temperature, scale, bias, bank,
                                    head, neighbors, posterior), head, neighbors, True)
            methods[method] = predictive
            selected_method = method.removesuffix("-forced") + "-selected"
            methods[selected_method] = predictive if enabled else (
                evaluate_base(rows, test, temperature, scale, bias), head, neighbors, False)
        for method, values in methods.items():
            if len(values) == 3:
                metrics, head, neighbors = values
                enabled = False
            else:
                metrics, head, neighbors, enabled = values
            output.append({
                "quant": name, "fold": fold,
                "test_chunks": "+".join(map(str, test_chunks)),
                "inner_validation_chunks": "+".join(map(str, inner_validation_chunks)),
                "train_positions": int(train.sum()), "test_positions": int(test.sum()),
                "temperature": temperature, "method": method,
                "selected_head": head, "selected_neighbors": neighbors,
                "selected_enabled": enabled,
                "scale": scale, **metrics,
            })
        print(
            f"finished {name} fold {fold}: mean={configs[False]}, "
            f"posterior={configs[True]}", flush=True)
    return output


def write_report(path: Path, rows: list[dict], reference_label: str) -> None:
    identity = np.mean([row["kl"] for row in rows if row["method"] == "identity"])
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Conditional empirical posterior sampler",
        "",
        f"The target is {reference_label}. Quant-only entropy, mass, and top-gap features",
        "retrieve calibration rows with similar candidate head shapes. The conditional-mean",
        "method applies their mean rank-wise residual. The posterior-predictive method instead",
        "averages the softmax distributions produced by their correlated residual samples.",
        "Head size and neighbor count are selected separately for each method on an inner",
        "chunk-held-out split. Selected variants may reject the transform and retain the",
        "static-bias baseline; forced variants show the best non-null inner configuration.",
        "All reported metrics come from untouched outer chunks.",
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
        "| Fold | Method | Head | Neighbors | Enabled |", "|---:|---|---:|---:|:---:|",
    ]
    for row in rows:
        if row["method"] in ("conditional-mean-selected", "posterior-predictive-selected"):
            lines.append(
                f"| {row['fold']} | {row['method']} | {row['selected_head']} | "
                f"{row['selected_neighbors']} | {'yes' if row['selected_enabled'] else 'no'} |")
    lines += ["", "Fold-level values are in `posterior.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--heads", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--neighbors", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(args.reference, args.candidate, args.candidate_name, args.temperature,
                   args.folds, args.heads, args.neighbors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "posterior.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir / "POSTERIOR.md", rows, args.reference_label)
    print(f"wrote {args.output_dir / 'POSTERIOR.md'}")


if __name__ == "__main__":
    main()
