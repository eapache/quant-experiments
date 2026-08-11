#!/usr/bin/env python3
"""Test whether the held-out low-rank oracle beats matched null subspaces."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_low_rank import (BasisModel, base_and_residual, evaluate, fit_basis,
                              head_positions)
from analyze_sparse import fit_temperature, fit_token_bias, load_sparse


@dataclass
class FoldResult:
    metrics: list[dict]
    spectrum: list[dict]


def randomized_svd_basis(matrix: np.ndarray, token_to_column: np.ndarray,
                         components: int, seed: int) -> BasisModel:
    """Return the leading right singular vectors using the experiment's SVD scheme."""
    rank = min(components, matrix.shape[0] - 1, matrix.shape[1])
    if rank <= 0:
        return BasisModel(token_to_column, np.empty((0, matrix.shape[1]), dtype=np.float32),
                          np.empty(0, dtype=np.float32))
    sketch_rank = min(rank + 12, matrix.shape[0] - 1, matrix.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], sketch_rank), dtype=np.float32)
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    _, singular_values, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
    return BasisModel(token_to_column.copy(), vt[:rank].astype(np.float32),
                      singular_values[:rank].astype(np.float32))


def training_residual_matrix(rows, selected: np.ndarray, model: BasisModel, scale: float,
                             bias: np.ndarray, temperature: float, head: int) -> np.ndarray:
    """Recreate the sparse head-residual matrix on the learned basis vocabulary."""
    indices = np.flatnonzero(selected)
    matrix = np.zeros((len(indices), model.basis.shape[1]), dtype=np.float32)
    for output_row, index in enumerate(indices):
        row = rows[index]
        positions = head_positions(row, head)
        _, _, residual = base_and_residual(row, scale, bias, temperature)
        columns = model.token_to_column[row.token_ids[positions]]
        valid = columns >= 0
        matrix[output_row, columns[valid]] = residual[positions[valid]]
    matrix -= matrix.mean(axis=0, keepdims=True)
    return matrix


def gaussian_basis(model: BasisModel, components: int, seed: int) -> BasisModel:
    """A target-independent Haar-like random subspace on the same token vocabulary."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((model.basis.shape[1], components), dtype=np.float32)
    q, _ = np.linalg.qr(values, mode="reduced")
    return BasisModel(model.token_to_column.copy(), q.T.astype(np.float32),
                      np.ones(components, dtype=np.float32))


def permuted_basis(model: BasisModel, components: int, seed: int) -> BasisModel:
    """Preserve learned directions and spectrum but break their token alignment."""
    permutation = np.random.default_rng(seed).permutation(model.basis.shape[1])
    return BasisModel(model.token_to_column.copy(),
                      model.basis[:components, permutation].copy(),
                      model.singular_values[:components].copy())


def sign_null_basis(matrix: np.ndarray, model: BasisModel, components: int,
                    seed: int) -> BasisModel:
    """Preserve every residual magnitude and zero but destroy signed covariance."""
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=matrix.shape, dtype=np.int8)
    signs = signs * 2 - 1
    null_matrix = matrix * signs
    null_matrix -= null_matrix.mean(axis=0, keepdims=True)
    return randomized_svd_basis(null_matrix, model.token_to_column, components, seed + 1)


def metric_row(fold: int, basis_kind: str, seed: int | None, components: int,
               identity_kl: float, baseline_kl: float, values: dict) -> dict:
    return {
        "fold": fold,
        "basis": basis_kind,
        "seed": "" if seed is None else seed,
        "components": components,
        "identity_kl": identity_kl,
        "baseline_kl": baseline_kl,
        "oracle_kl": values["kl"],
        "recovered_from_identity": 1.0 - values["kl"] / identity_kl,
        "recovered_from_baseline": 1.0 - values["kl"] / baseline_kl,
        "js": values["js"],
        "tv": values["tv"],
        "top1_agreement": values["top1_agreement"],
        "top10_overlap": values["top10_overlap"],
    }


def analyze_fold(reference_path: Path, candidate_path: Path, fold: int, folds: int,
                 temperature: float, head: int, components_grid: list[int],
                 control_components: int, control_seeds: int) -> FoldResult:
    rows, chunks, n_vocab, n_chunks = load_sparse(reference_path, candidate_path)
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    test_chunks = list(range(fold * fold_size, (fold + 1) * fold_size))
    test = np.isin(chunks, test_chunks)
    train = ~test
    scale = fit_temperature(rows, train, temperature)
    bias = fit_token_bias(rows, train, n_vocab, scale, temperature)
    zero_bias = np.zeros(n_vocab, dtype=np.float32)
    identity = evaluate(rows, test, temperature, 1.0, zero_bias)
    baseline = evaluate(rows, test, temperature, scale, bias)

    metrics = [metric_row(fold, "identity", None, 0, identity["kl"], baseline["kl"], identity),
               metric_row(fold, "temperature-token", None, 0, identity["kl"],
                          baseline["kl"], baseline)]
    learned_by_rank = {}
    for components in components_grid:
        learned_by_rank[components] = fit_basis(
            rows, train, n_vocab, scale, bias, temperature, head, components,
            seed=2000 + fold)
        values = evaluate(rows, test, temperature, scale, bias,
                          learned_by_rank[components], components, oracle=True)
        metrics.append(metric_row(fold, "learned-residual-pca", None, components,
                                  identity["kl"], baseline["kl"], values))

    learned = learned_by_rank[control_components]
    matrix = training_residual_matrix(
        rows, train, learned, scale, bias, temperature, head)
    spectrum: list[dict] = []
    null_singular_values = []
    for repeat in range(control_seeds):
        seed = 100_000 + 1000 * fold + repeat
        gaussian = gaussian_basis(learned, max(components_grid), seed)
        for components in components_grid:
            values = evaluate(rows, test, temperature, scale, bias, gaussian,
                              components, oracle=True)
            metrics.append(metric_row(fold, "gaussian-random", repeat, components,
                                      identity["kl"], baseline["kl"], values))
        controls = {
            "token-permuted-pca": permuted_basis(learned, control_components, seed + 100),
            "sign-randomized-pca": sign_null_basis(
                matrix, learned, control_components, seed + 200),
        }
        null_singular_values.append(controls["sign-randomized-pca"].singular_values)
        for name, control in controls.items():
            values = evaluate(rows, test, temperature, scale, bias, control,
                              control_components, oracle=True)
            metrics.append(metric_row(fold, name, repeat, control_components,
                                      identity["kl"], baseline["kl"], values))
        print(f"fold {fold}: finished null repeat {repeat + 1}/{control_seeds}", flush=True)

    null_singular = np.stack(null_singular_values)
    for component in range(control_components):
        spectrum.append({
            "fold": fold,
            "component": component + 1,
            "learned_singular_value": learned.singular_values[component],
            "sign_null_singular_mean": null_singular[:, component].mean(),
            "sign_null_singular_sd": null_singular[:, component].std(ddof=1)
            if control_seeds > 1 else 0.0,
            "learned_to_null_ratio": learned.singular_values[component]
            / null_singular[:, component].mean(),
        })
    print(f"finished fold {fold}", flush=True)
    return FoldResult(metrics, spectrum)


def aggregate_by_seed(rows: list[dict], basis: str, components: int) -> np.ndarray:
    selected = [row for row in rows
                if row["basis"] == basis and int(row["components"]) == components]
    seeds = sorted({int(row["seed"]) for row in selected})
    return np.array([
        np.mean([row["oracle_kl"] for row in selected if int(row["seed"]) == seed])
        for seed in seeds
    ])


def distribution_support_summary(reference_path: Path,
                                 candidate_path: Path) -> list[dict]:
    rows, _, _, _ = load_sparse(reference_path, candidate_path)
    output = []
    for label, attribute in (("BF16", "reference"), ("Q2", "candidate")):
        effective, nucleus_95, top16, top32 = [], [], [], []
        for row in rows:
            probabilities = np.exp(getattr(row, attribute).astype(np.float64)
                                   - float(np.max(getattr(row, attribute))))
            probabilities /= probabilities.sum()
            ordered = np.sort(probabilities)[::-1]
            cumulative = np.cumsum(ordered)
            effective.append(float(np.exp(-np.dot(probabilities, np.log(probabilities)))))
            nucleus_95.append(int(np.searchsorted(cumulative, 0.95) + 1))
            top16.append(float(ordered[:16].sum()))
            top32.append(float(ordered[:32].sum()))
        output.append({
            "distribution": label,
            "effective_support_mean": np.mean(effective),
            "effective_support_median": np.median(effective),
            "nucleus_95_mean": np.mean(nucleus_95),
            "nucleus_95_median": np.median(nucleus_95),
            "top16_mass_mean": np.mean(top16),
            "top32_mass_mean": np.mean(top32),
        })
    return output


def write_report(path: Path, rows: list[dict], spectrum: list[dict], support: list[dict],
                 control_components: int, control_seeds: int) -> None:
    identity = np.mean([row["oracle_kl"] for row in rows if row["basis"] == "identity"])
    baseline = np.mean([
        row["oracle_kl"] for row in rows if row["basis"] == "temperature-token"])
    learned_rows = [row for row in rows if row["basis"] == "learned-residual-pca"]
    ranks = sorted({int(row["components"]) for row in learned_rows})
    lines = [
        "# Controlled low-rank residual diagnostic", "",
        "Every basis is constructed without the outer test chunks. The oracle then receives",
        "the held-out BF16 row and optimizes the same number of per-row amplitudes for learned",
        "and control bases. Controls are Gaussian random subspaces, token-permuted learned PCA",
        "directions, and PCA after independently randomizing every training residual sign.", "",
        "## Learned basis versus Gaussian-null rank curve", "",
        "| Directions | Learned KL | Gaussian-null KL | Learned advantage | Learned recovery | Top-1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank in ranks:
        selected = [row for row in learned_rows if int(row["components"]) == rank]
        kl = np.mean([row["oracle_kl"] for row in selected])
        gaussian = [row for row in rows
                    if row["basis"] == "gaussian-random"
                    and int(row["components"]) == rank]
        gaussian_kl = np.mean([row["oracle_kl"] for row in gaussian])
        lines.append(
            f"| {rank} | {kl:.7f} | {gaussian_kl:.7f} | {gaussian_kl - kl:.7f} | "
            f"{1.0 - kl / identity:.2%} | "
            f"{np.mean([row['top1_agreement'] for row in selected]):.1%} |")

    lines += ["", f"## Rank-{control_components} matched controls", "",
              "Control SD/range is across seeds after averaging the four outer folds.", "",
              "| Basis | Oracle KL | Seed SD | Seed range | Recovered from raw |",
              "|---|---:|---:|---:|---:|"]
    learned = [row for row in learned_rows if int(row["components"]) == control_components]
    learned_kl = np.mean([row["oracle_kl"] for row in learned])
    lines.append(f"| learned-residual-pca | {learned_kl:.7f} | — | — | "
                 f"{1.0 - learned_kl / identity:.2%} |")
    for basis in ("gaussian-random", "token-permuted-pca", "sign-randomized-pca"):
        seed_values = aggregate_by_seed(rows, basis, control_components)
        kl = seed_values.mean()
        lines.append(
            f"| {basis} | {kl:.7f} | {seed_values.std(ddof=1):.7f} | "
            f"[{seed_values.min():.7f}, {seed_values.max():.7f}] | "
            f"{1.0 - kl / identity:.2%} |")

    lines += ["", "## Distribution concentration", "",
              "| Distribution | Effective support mean | Effective support median | Mean top-16 mass | Mean top-32 mass |",
              "|---|---:|---:|---:|---:|"]
    for row in support:
        lines.append(
            f"| {row['distribution']} | {row['effective_support_mean']:.1f} | "
            f"{row['effective_support_median']:.1f} | {row['top16_mass_mean']:.1%} | "
            f"{row['top32_mass_mean']:.1%} |")

    ratios = np.array([row["learned_to_null_ratio"] for row in spectrum])
    first = np.mean([row["learned_to_null_ratio"] for row in spectrum
                     if int(row["component"]) == 1])
    last = np.mean([row["learned_to_null_ratio"] for row in spectrum
                    if int(row["component"]) == control_components])
    strongest_control = min(
        aggregate_by_seed(rows, basis, control_components).min()
        for basis in ("gaussian-random", "token-permuted-pca", "sign-randomized-pca"))
    all_folds_better = all(
        next(row["oracle_kl"] for row in learned
             if int(row["fold"]) == fold
             and int(row["components"]) == control_components)
        < min(row["oracle_kl"] for row in rows
              if row["basis"] in ("gaussian-random", "token-permuted-pca",
                                  "sign-randomized-pca")
              and int(row["fold"]) == fold
              and int(row["components"]) == control_components)
        for fold in sorted({int(row["fold"]) for row in learned})
    )
    gaussian_kl = aggregate_by_seed(
        rows, "gaussian-random", control_components).mean()
    learned_reduction = identity - learned_kl
    learned_specific = gaussian_kl - learned_kl
    fold_advantages = []
    for fold in sorted({int(row["fold"]) for row in learned}):
        learned_fold = next(row["oracle_kl"] for row in learned
                            if int(row["fold"]) == fold)
        gaussian_fold = np.mean([
            row["oracle_kl"] for row in rows
            if row["basis"] == "gaussian-random"
            and int(row["components"]) == control_components
            and int(row["fold"]) == fold])
        fold_advantages.append(gaussian_fold - learned_fold)
    fold_advantages = np.asarray(fold_advantages)
    # t(3), 97.5%; the four outer blocks, not control seeds, are the data units.
    advantage_half_width = 3.182446 * fold_advantages.std(ddof=1) / np.sqrt(
        len(fold_advantages))
    lines += [
        "", "## Interpretation", "",
        f"The learned rank-{control_components} basis has KL {learned_kl:.7f}; the best",
        f"aggregate control seed has KL {strongest_control:.7f}. It beats every matched",
        f"control in every outer fold: {'yes' if all_folds_better else 'no'}.",
        f"Learned singular values are {first:.2f}× the sign-null value for component 1 and",
        f"{last:.2f}× for component {control_components} (all-component mean {ratios.mean():.2f}×).",
        f"However, the learned basis improves over the mean Gaussian oracle by only",
        f"{learned_specific:.7f} KL, {learned_specific / learned_reduction:.1%} of its total",
        f"oracle reduction. Its four-block t interval is [{learned_specific - advantage_half_width:.7f},",
        f"{learned_specific + advantage_half_width:.7f}], so the learned advantage is not",
        "independently established at the document/workload level.",
        "The rank curve has no sharp low-dimensional cutoff and the Gaussian curve tracks it",
        "closely. Thus the results suggest some transferable residual covariance, but the prior",
        "57% figure mostly measures generic per-row oracle flexibility on a",
        "concentrated distribution; it is not evidence that the residual is intrinsically",
        "16-dimensional. None of these controls make the oracle deployable or show that its",
        "amplitudes are predictable from quant-only inputs.",
        "", f"Each null family used {control_seeds} fixed seeds. Fold-level metrics are in",
        "`low_rank_structure.csv`; singular-value comparisons are in `low_rank_spectrum.csv`,",
        "and concentration diagnostics are in `distribution_support.csv`.", "",
    ]
    path.write_text("\n".join(lines))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--control-components", type=int, default=16)
    parser.add_argument("--control-seeds", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.control_components not in args.components:
        parser.error("--control-components must appear in --components")

    calls = [(args.reference, args.candidate, fold, args.folds, args.temperature,
              args.head, args.components, args.control_components, args.control_seeds)
             for fold in range(args.folds)]
    if args.jobs == 1:
        results = [analyze_fold(*call) for call in calls]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(analyze_fold, *call) for call in calls]
            results = [future.result() for future in futures]
    rows = [row for result in results for row in result.metrics]
    spectrum = [row for result in results for row in result.spectrum]
    support = distribution_support_summary(args.reference, args.candidate)
    rows.sort(key=lambda row: (int(row["fold"]), row["basis"], str(row["seed"]),
                               int(row["components"])))
    spectrum.sort(key=lambda row: (int(row["fold"]), int(row["component"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "low_rank_structure.csv", rows)
    write_csv(args.output_dir / "low_rank_spectrum.csv", spectrum)
    write_csv(args.output_dir / "distribution_support.csv", support)
    write_report(args.output_dir / "LOW_RANK_STRUCTURE.md", rows, spectrum, support,
                 args.control_components, args.control_seeds)
    print(f"wrote {args.output_dir / 'LOW_RANK_STRUCTURE.md'}")


if __name__ == "__main__":
    main()
