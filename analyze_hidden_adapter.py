#!/usr/bin/env python3
"""Nested held-out test of a Q2-hidden-state residual amplitude adapter."""

from __future__ import annotations

import argparse
import csv
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_low_rank import (BasisModel, evaluate, fit_basis, oracle_coefficients,
                              row_basis)
from analyze_sparse import PairedRow, fit_temperature, fit_token_bias, load_sparse


@dataclass
class KernelTraining:
    feature_mean: np.ndarray
    feature_scale: float
    standardized: np.ndarray
    target_mean: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_targets: np.ndarray


def read_capture_header(path: Path) -> tuple[int, int, int, np.ndarray]:
    with path.open("rb") as stream:
        if stream.read(8) != b"_logits_":
            raise ValueError(f"{path} is not a llama.cpp logits capture")
        n_ctx, n_vocab, n_chunks = struct.unpack("<Iii", stream.read(12))
        tokens = np.fromfile(stream, dtype="<i4", count=n_ctx * n_chunks)
    if len(tokens) != n_ctx * n_chunks:
        raise ValueError(f"truncated token header in {path}")
    return n_ctx, n_vocab, n_chunks, tokens


def load_hidden(path: Path, capture_path: Path) -> np.memmap:
    with path.open("rb") as stream:
        if stream.read(8) != b"_hidden_":
            raise ValueError(f"{path} is not a hidden-state capture")
        values = stream.read(24)
        if len(values) != 24:
            raise ValueError(f"truncated hidden-state header in {path}")
        version, n_ctx, n_embd, n_chunks, first, rows_per_chunk = struct.unpack(
            "<6I", values)
        tokens = np.fromfile(stream, dtype="<i4", count=n_ctx * n_chunks)
    if version != 1:
        raise ValueError(f"unsupported hidden-state version {version}")
    capture_ctx, _, capture_chunks, capture_tokens = read_capture_header(capture_path)
    if n_ctx != capture_ctx or n_chunks != capture_chunks:
        raise ValueError("hidden-state and logits capture dimensions differ")
    if first != n_ctx // 2 or rows_per_chunk != n_ctx - first - 1:
        raise ValueError("hidden-state row range does not match perplexity evaluation")
    if not np.array_equal(tokens, capture_tokens):
        raise ValueError("hidden-state and logits capture tokens differ")
    offset = 32 + n_ctx * n_chunks * np.dtype("<i4").itemsize
    shape = (n_chunks * rows_per_chunk, n_embd)
    expected = offset + np.prod(shape) * np.dtype("<f4").itemsize
    if path.stat().st_size != expected:
        raise ValueError(
            f"hidden-state file has {path.stat().st_size} bytes; expected {expected}")
    return np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=shape)


def prepare_kernel(features: np.ndarray, targets: np.ndarray) -> KernelTraining:
    feature_mean = np.asarray(features.mean(axis=0), dtype=np.float32)
    centered = np.asarray(features, dtype=np.float32) - feature_mean
    feature_scale = float(np.sqrt(np.mean(centered.astype(np.float64) ** 2)))
    if not np.isfinite(feature_scale) or feature_scale < 1e-8:
        feature_scale = 1.0
    standardized = np.asarray(centered / feature_scale, dtype=np.float32)
    target_mean = np.asarray(targets.mean(axis=0), dtype=np.float64)
    centered_targets = np.asarray(targets, dtype=np.float64) - target_mean
    kernel = np.asarray(standardized @ standardized.T, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_targets = eigenvectors.T @ centered_targets
    return KernelTraining(feature_mean, feature_scale, standardized, target_mean,
                          eigenvalues, eigenvectors, projected_targets)


def predict_kernel(model: KernelTraining, alpha: float,
                   features: np.ndarray) -> np.ndarray:
    weights = model.eigenvectors @ (
        model.projected_targets / (model.eigenvalues[:, None] + alpha))
    standardized = (np.asarray(features, dtype=np.float32) - model.feature_mean)
    standardized /= model.feature_scale
    cross_kernel = np.asarray(standardized @ model.standardized.T, dtype=np.float64)
    return model.target_mean + cross_kernel @ weights


def target_coefficients(rows: list[PairedRow], indices: np.ndarray, basis: BasisModel,
                        components: int, scale: float, bias: np.ndarray,
                        temperature: float) -> np.ndarray:
    return np.stack([
        oracle_coefficients(rows[index], basis, components, scale, bias, temperature)
        for index in indices
    ]).astype(np.float32)


def evaluate_predictions(rows: list[PairedRow], indices: np.ndarray, temperature: float,
                         scale: float, bias: np.ndarray, basis: BasisModel,
                         components: int, coefficients: np.ndarray) -> dict[str, float]:
    kl, js, tv, top1, top10 = [], [], [], [], []
    for coefficient_row, index in zip(coefficients, indices):
        row = rows[index]
        reference_scaled = row.reference.astype(np.float64) / temperature
        maximum = float(reference_scaled.max())
        log_p = reference_scaled - (maximum + np.log(np.exp(reference_scaled - maximum).sum()))
        p = np.exp(log_p)
        corrected = scale * row.candidate + bias[row.token_ids]
        corrected = corrected + row_basis(row, basis, components) @ coefficient_row
        candidate_scaled = corrected.astype(np.float64) / temperature
        maximum = float(candidate_scaled.max())
        log_q = candidate_scaled - (maximum + np.log(np.exp(candidate_scaled - maximum).sum()))
        q = np.exp(log_q)
        kl.append(float(np.dot(p, log_p - log_q)))
        midpoint = 0.5 * (p + q)
        js.append(float(0.5 * np.dot(p, log_p - np.log(midpoint))
                        + 0.5 * np.dot(q, log_q - np.log(midpoint))))
        tv.append(float(0.5 * np.abs(p - q).sum()))
        ref_top = np.argpartition(row.reference, -10)[-10:]
        corrected_top = np.argpartition(corrected, -10)[-10:]
        top1.append(int(np.argmax(row.reference) == np.argmax(corrected)))
        top10.append(np.intersect1d(ref_top, corrected_top, assume_unique=True).size / 10.0)
    return {
        "kl": float(np.mean(kl)), "js": float(np.mean(js)), "tv": float(np.mean(tv)),
        "top1_agreement": float(np.mean(top1)), "top10_overlap": float(np.mean(top10)),
    }


def select_hyperparameters(rows: list[PairedRow], hidden: np.ndarray,
                           inner_train: np.ndarray, inner_validation: np.ndarray,
                           n_vocab: int, temperature: float, head: int,
                           components_grid: list[int], alpha_grid: list[float],
                           seed: int) -> tuple[int, float, int, float]:
    train_indices = np.flatnonzero(inner_train)
    validation_indices = np.flatnonzero(inner_validation)
    scale = fit_temperature(rows, inner_train, temperature)
    bias = fit_token_bias(rows, inner_train, n_vocab, scale, temperature)
    basis = fit_basis(rows, inner_train, n_vocab, scale, bias, temperature, head,
                      max(components_grid), seed)
    baseline = evaluate(rows, inner_validation, temperature, scale, bias)["kl"]
    best = (baseline, 0, alpha_grid[0])
    best_forced = (float("inf"), 0, alpha_grid[0])
    for components in components_grid:
        if components == 0 or components > len(basis.basis):
            continue
        targets = target_coefficients(rows, train_indices, basis, components, scale, bias,
                                      temperature)
        kernel = prepare_kernel(hidden[train_indices], targets)
        for alpha in alpha_grid:
            predictions = predict_kernel(kernel, alpha, hidden[validation_indices])
            value = evaluate_predictions(rows, validation_indices, temperature, scale, bias,
                                         basis, components, predictions)["kl"]
            candidate = (value, components, alpha)
            if candidate < best:
                best = candidate
            if candidate < best_forced:
                best_forced = candidate
    return int(best[1]), float(best[2]), int(best_forced[1]), float(best_forced[2])


def analyze(reference_path: Path, candidate_path: Path, hidden_path: Path,
            temperature: float, folds: int, head: int, components_grid: list[int],
            alpha_grid: list[float]) -> list[dict]:
    rows, chunks, n_vocab, n_chunks = load_sparse(reference_path, candidate_path)
    hidden = load_hidden(hidden_path, candidate_path)
    if len(hidden) != len(rows):
        raise ValueError(f"{len(hidden)} hidden states do not align with {len(rows)} rows")
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    output: list[dict] = []
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
        selected_components, selected_alpha, forced_components, forced_alpha = select_hyperparameters(
            rows, hidden, inner_train, inner_validation, n_vocab, temperature, head,
            components_grid, alpha_grid, seed=3000 + fold)

        scale = fit_temperature(rows, train, temperature)
        bias = fit_token_bias(rows, train, n_vocab, scale, temperature)
        basis = fit_basis(rows, train, n_vocab, scale, bias, temperature, head,
                          max_components, seed=4000 + fold)
        methods = {
            "identity": evaluate(rows, test, temperature, 1.0, zero_bias),
            "target-temperature": evaluate(rows, test, temperature, scale, zero_bias),
            "target-temperature+token": evaluate(rows, test, temperature, scale, bias),
        }
        if selected_components:
            train_indices = np.flatnonzero(train)
            test_indices = np.flatnonzero(test)
            targets = target_coefficients(rows, train_indices, basis, selected_components,
                                          scale, bias, temperature)
            kernel = prepare_kernel(hidden[train_indices], targets)
            predictions = predict_kernel(kernel, selected_alpha, hidden[test_indices])
            methods["hidden-adapter"] = evaluate_predictions(
                rows, test_indices, temperature, scale, bias, basis, selected_components,
                predictions)
        else:
            methods["hidden-adapter"] = methods["target-temperature+token"]
        forced_train_indices = np.flatnonzero(train)
        forced_test_indices = np.flatnonzero(test)
        forced_targets = target_coefficients(
            rows, forced_train_indices, basis, forced_components, scale, bias, temperature)
        forced_kernel = prepare_kernel(hidden[forced_train_indices], forced_targets)
        forced_predictions = predict_kernel(
            forced_kernel, forced_alpha, hidden[forced_test_indices])
        methods["hidden-adapter-forced"] = evaluate_predictions(
            rows, forced_test_indices, temperature, scale, bias, basis, forced_components,
            forced_predictions)
        methods[f"low-rank-oracle-{max_components}"] = evaluate(
            rows, test, temperature, scale, bias, basis, len(basis.basis), oracle=True)

        for method, metrics in methods.items():
            output.append({
                "fold": fold, "test_chunks": "+".join(map(str, test_chunks)),
                "inner_validation_chunks": "+".join(map(str, inner_validation_chunks)),
                "train_positions": int(train.sum()), "test_positions": int(test.sum()),
                "temperature": temperature, "head": head, "method": method,
                "selected_components": selected_components,
                "selected_alpha": selected_alpha, "forced_components": forced_components,
                "forced_alpha": forced_alpha, "scale": scale, **metrics,
            })
        print(f"finished fold {fold}: selected k={selected_components}, alpha={selected_alpha:g}; "
              f"forced k={forced_components}, alpha={forced_alpha:g}", flush=True)
    return output


def write_report(path: Path, rows: list[dict], reference_label: str,
                 candidate_label: str, hidden_dimensions: int) -> None:
    identity = np.mean([row["kl"] for row in rows if row["method"] == "identity"])
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Final-hidden-state residual adapter",
        "",
        f"The target is {reference_label}; inputs are the {hidden_dimensions}-dimensional normalized",
        f"final hidden states from {candidate_label}. The adapter is a kernel-form ridge map to",
        "oracle amplitudes of head-restricted residual directions. Complete context chunks are",
        "held out, and both direction count and ridge penalty include a null option and are selected",
        "only on an inner chunk split.",
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
    selected_rows = [row for row in rows if row["method"] == "hidden-adapter"]
    forced_kl = np.mean([
        row["kl"] for row in rows if row["method"] == "hidden-adapter-forced"])
    baseline_kl = np.mean([
        row["kl"] for row in rows if row["method"] == "target-temperature+token"])
    null_folds = sum(row["selected_components"] == 0 for row in selected_rows)
    lines += [
        "",
        f"The null adapter was selected in {null_folds}/{len(selected_rows)} folds. Forced use",
        f"changed KL from {baseline_kl:.7f} to {forced_kl:.7f}; it is a diagnostic only, not a",
        "deployable correction. This first pass still uses only 2,016 paired positions, so it",
        "does not replace the planned larger-corpus test.",
    ]
    lines += ["", "## Inner-selected capacity", "",
              "| Fold | Selected directions | Selected alpha | Forced directions | Forced alpha |",
              "|---:|---:|---:|---:|---:|"]
    for row in [row for row in rows if row["method"] == "hidden-adapter"]:
        lines.append(f"| {row['fold']} | {row['selected_components']} | "
                     f"{row['selected_alpha']:g} | {row['forced_components']} | "
                     f"{row['forced_alpha']:g} |")
    lines += ["", "Fold-level values are in `hidden_adapter.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--reference-label", default="BF16")
    parser.add_argument("--candidate-label", default="Q2_K_XL")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--components", type=int, nargs="+", default=[0, 4, 8, 16])
    parser.add_argument("--ridge-alpha", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(args.reference, args.candidate, args.hidden, args.temperature,
                   args.folds, args.head, args.components, args.ridge_alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "hidden_adapter.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hidden_dimensions = load_hidden(args.hidden, args.candidate).shape[1]
    write_report(args.output_dir / "HIDDEN_ADAPTER.md", rows, args.reference_label,
                 args.candidate_label, hidden_dimensions)
    print(f"wrote {args.output_dir / 'HIDDEN_ADAPTER.md'}")


if __name__ == "__main__":
    main()
