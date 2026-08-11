#!/usr/bin/env python3
"""Evaluate prose-trained learned and null subspaces with an oracle on code."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_low_rank import evaluate, fit_basis
from analyze_low_rank_structure import (gaussian_basis, metric_row, permuted_basis,
                                        sign_null_basis, training_residual_matrix)
from analyze_sparse import fit_temperature, fit_token_bias, load_sparse


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(train_reference: Path, train_candidate: Path, ood_reference: Path,
            ood_candidate: Path, temperature: float, head: int,
            components_grid: list[int], control_components: int,
            control_seeds: int) -> tuple[list[dict], list[dict]]:
    train_rows, _, n_vocab, _ = load_sparse(train_reference, train_candidate)
    test_rows, test_chunks, test_vocab, n_test_chunks = load_sparse(
        ood_reference, ood_candidate)
    if test_vocab != n_vocab:
        raise ValueError(f"training vocabulary {n_vocab} != OOD vocabulary {test_vocab}")
    train = np.ones(len(train_rows), dtype=bool)
    test = np.ones(len(test_rows), dtype=bool)
    scale = fit_temperature(train_rows, train, temperature)
    bias = fit_token_bias(train_rows, train, n_vocab, scale, temperature)
    zero_bias = np.zeros(n_vocab, dtype=np.float32)
    identity = evaluate(test_rows, test, temperature, 1.0, zero_bias)
    baseline = evaluate(test_rows, test, temperature, scale, bias)
    rows = [metric_row(0, "identity", None, 0, identity["kl"], baseline["kl"], identity),
            metric_row(0, "frozen-temperature-token", None, 0, identity["kl"],
                       baseline["kl"], baseline)]

    learned_by_rank = {}
    for components in components_grid:
        learned_by_rank[components] = fit_basis(
            train_rows, train, n_vocab, scale, bias, temperature, head, components,
            seed=7000)
        values = evaluate(test_rows, test, temperature, scale, bias,
                          learned_by_rank[components], components, oracle=True)
        rows.append(metric_row(0, "learned-residual-pca", None, components,
                               identity["kl"], baseline["kl"], values))

    learned = learned_by_rank[control_components]
    matrix = training_residual_matrix(
        train_rows, train, learned, scale, bias, temperature, head)
    gaussian_models = []
    for repeat in range(control_seeds):
        seed = 300_000 + repeat
        gaussian = gaussian_basis(learned, max(components_grid), seed)
        gaussian_models.append(gaussian)
        for components in components_grid:
            values = evaluate(test_rows, test, temperature, scale, bias, gaussian,
                              components, oracle=True)
            rows.append(metric_row(0, "gaussian-random", repeat, components,
                                   identity["kl"], baseline["kl"], values))
        controls = {
            "token-permuted-pca": permuted_basis(
                learned, control_components, seed + 100),
            "sign-randomized-pca": sign_null_basis(
                matrix, learned, control_components, seed + 200),
        }
        for name, control in controls.items():
            values = evaluate(test_rows, test, temperature, scale, bias, control,
                              control_components, oracle=True)
            rows.append(metric_row(0, name, repeat, control_components,
                                   identity["kl"], baseline["kl"], values))
        print(f"finished OOD null repeat {repeat + 1}/{control_seeds}", flush=True)

    chunk_rows = []
    for chunk in range(n_test_chunks):
        selected = test_chunks == chunk
        learned_values = evaluate(
            test_rows, selected, temperature, scale, bias, learned,
            control_components, oracle=True)
        gaussian_values = [
            evaluate(test_rows, selected, temperature, scale, bias, model,
                     control_components, oracle=True)["kl"]
            for model in gaussian_models
        ]
        chunk_rows.append({
            "chunk": chunk,
            "positions": int(selected.sum()),
            "learned_oracle_kl": learned_values["kl"],
            "gaussian_oracle_kl_mean": np.mean(gaussian_values),
            "learned_advantage": np.mean(gaussian_values) - learned_values["kl"],
        })
    return rows, chunk_rows


def write_report(path: Path, rows: list[dict], chunk_rows: list[dict],
                 components_grid: list[int], control_components: int) -> None:
    identity = next(row["oracle_kl"] for row in rows if row["basis"] == "identity")
    lines = [
        "# Frozen prose-to-code low-rank structure check", "",
        "Every subspace, scalar, and static token bias is fitted on all 2,016 prose positions.",
        "The code capture is loaded only for evaluation. BF16 still supplies per-row oracle",
        "amplitudes to learned and null bases, so this diagnoses structure rather than a",
        "deployable correction.", "", "## Learned versus Gaussian-null rank curve", "",
        "| Directions | Learned KL | Gaussian KL | Learned advantage | Learned recovery |",
        "|---:|---:|---:|---:|---:|",
    ]
    for components in components_grid:
        learned = next(row["oracle_kl"] for row in rows
                       if row["basis"] == "learned-residual-pca"
                       and int(row["components"]) == components)
        gaussian = np.mean([
            row["oracle_kl"] for row in rows
            if row["basis"] == "gaussian-random"
            and int(row["components"]) == components])
        lines.append(f"| {components} | {learned:.7f} | {gaussian:.7f} | "
                     f"{gaussian - learned:.7f} | {1.0 - learned / identity:.2%} |")

    lines += ["", f"## Rank-{control_components} controls", "",
              "| Basis | Oracle KL | Recovered from raw |", "|---|---:|---:|"]
    for basis in ("learned-residual-pca", "gaussian-random", "token-permuted-pca",
                  "sign-randomized-pca"):
        selected = [row["oracle_kl"] for row in rows
                    if row["basis"] == basis
                    and int(row["components"]) == control_components]
        kl = np.mean(selected)
        lines.append(f"| {basis} | {kl:.7f} | {1.0 - kl / identity:.2%} |")

    differences = np.array([row["learned_advantage"] for row in chunk_rows])
    half_width = 2.039513 * differences.std(ddof=1) / np.sqrt(len(differences))
    lines += [
        "", "## Per-chunk learned advantage over Gaussian", "",
        f"Mean advantage is {differences.mean():.7f} KL with a chunk-level t interval",
        f"[{differences.mean() - half_width:.7f}, {differences.mean() + half_width:.7f}].",
        f"Learned PCA wins in {np.count_nonzero(differences > 0)}/{len(differences)} chunks.",
        "These chunks come from one reused code file, so the interval is descriptive rather",
        "than a document-level generalization interval.", "",
        "Aggregate metrics are in `frozen_low_rank_structure.csv`; chunk values are in",
        "`frozen_low_rank_structure_chunks.csv`.", "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--train-candidate", type=Path, required=True)
    parser.add_argument("--ood-reference", type=Path, required=True)
    parser.add_argument("--ood-candidate", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--control-components", type=int, default=16)
    parser.add_argument("--control-seeds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.control_components not in args.components:
        parser.error("--control-components must appear in --components")
    rows, chunk_rows = analyze(
        args.train_reference, args.train_candidate, args.ood_reference,
        args.ood_candidate, args.temperature, args.head, args.components,
        args.control_components, args.control_seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_low_rank_structure.csv", rows)
    write_csv(args.output_dir / "frozen_low_rank_structure_chunks.csv", chunk_rows)
    write_report(args.output_dir / "FROZEN_LOW_RANK_STRUCTURE.md", rows, chunk_rows,
                 args.components, args.control_components)
    print(f"wrote {args.output_dir / 'FROZEN_LOW_RANK_STRUCTURE.md'}")


if __name__ == "__main__":
    main()
