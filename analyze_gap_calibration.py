#!/usr/bin/env python3
"""Nested held-out test of monotonic rank-conditioned Q2 gap calibration."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_sparse import PairedRow, fit_temperature, fit_token_bias, load_sparse, softmax


@dataclass
class MonotonicMap:
    x: np.ndarray
    y: np.ndarray


@dataclass
class GapModel:
    head: int
    rank_bins: int
    mappings: list[MonotonicMap]


@dataclass(frozen=True)
class Config:
    head: int
    knots: int
    rank_bins: int
    strength: float
    enabled: bool = True


def top_order(logits: np.ndarray, exact: np.ndarray, count: int) -> np.ndarray:
    """Return the highest-logit positions that are exact in both captures."""
    positions = np.flatnonzero(exact)
    count = min(count, len(positions))
    if count == len(positions):
        return positions[np.argsort(-logits[positions])]
    chosen = positions[np.argpartition(logits[positions], -count)[-count:]]
    return chosen[np.argsort(-logits[chosen])]


def rank_group(rank: np.ndarray | int, head: int, rank_bins: int) -> np.ndarray | int:
    """Bucket a one-based lower-token rank into equally wide head regions."""
    return np.minimum(((np.asarray(rank) - 1) * rank_bins) // max(head - 1, 1),
                      rank_bins - 1)


def fit_monotonic(x: np.ndarray, y: np.ndarray, weight: np.ndarray,
                  knots: int) -> MonotonicMap:
    """Fit a weighted, piecewise-linear isotonic map with quantile pre-binning."""
    if not len(x):
        return MonotonicMap(np.array([0.0], dtype=np.float32),
                            np.array([0.0], dtype=np.float32))
    order = np.argsort(x, kind="stable")
    x, y, weight = x[order], y[order], weight[order]
    boundaries = np.linspace(0, len(x), min(knots, len(x)) + 1, dtype=int)
    bx, by, bw = [], [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        total = float(weight[start:stop].sum())
        if total <= 0.0:
            continue
        bx.append(float(np.dot(weight[start:stop], x[start:stop]) / total))
        by.append(float(np.dot(weight[start:stop], y[start:stop]) / total))
        bw.append(total)

    # Weighted pool-adjacent-violators algorithm.
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    pooled_w: list[float] = []
    for xx, yy, ww in zip(bx, by, bw):
        pooled_x.append(xx * ww)
        pooled_y.append(yy * ww)
        pooled_w.append(ww)
        while len(pooled_y) >= 2:
            left = pooled_y[-2] / pooled_w[-2]
            right = pooled_y[-1] / pooled_w[-1]
            if left <= right:
                break
            last_x = pooled_x.pop()
            last_y = pooled_y.pop()
            last_w = pooled_w.pop()
            pooled_x[-1] += last_x
            pooled_y[-1] += last_y
            pooled_w[-1] += last_w

    output_x = np.array([xx / ww for xx, ww in zip(pooled_x, pooled_w)], dtype=np.float32)
    output_y = np.array([yy / ww for yy, ww in zip(pooled_y, pooled_w)], dtype=np.float32)
    if len(output_x) > 1:
        # np.interp requires increasing x; pooled weighted means can coincide after clipping.
        keep = np.r_[output_x[1:] > output_x[:-1], True]
        output_x, output_y = output_x[keep], output_y[keep]
    return MonotonicMap(output_x, output_y)


def predict(mapping: MonotonicMap, values: np.ndarray) -> np.ndarray:
    return np.interp(values, mapping.x, mapping.y).astype(np.float32)


def training_samples(rows: list[PairedRow], selected: np.ndarray, scale: float,
                     bias: np.ndarray, temperature: float, head: int,
                     rank_bins: int, pairwise: bool) -> list[tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]]:
    samples: list[list[list[float]]] = [[[], [], []] for _ in range(rank_bins)]
    for index in np.flatnonzero(selected):
        row = rows[index]
        base = scale * row.candidate + bias[row.token_ids]
        order = top_order(base, row.exact, head)
        if len(order) < 2:
            continue
        p = softmax(row.reference / temperature)
        q = softmax(base / temperature)
        mass = p[order].astype(np.float64) + q[order].astype(np.float64)
        if pairwise:
            upper, lower = np.triu_indices(len(order), 1)
            groups = rank_group(lower, head, rank_bins)
            x = base[order[upper]] - base[order[lower]]
            y = row.reference[order[upper]] - row.reference[order[lower]]
            weight = np.sqrt(np.maximum(mass[upper] * mass[lower], 1e-12))
            for group in range(rank_bins):
                keep = groups == group
                samples[group][0].extend(x[keep])
                samples[group][1].extend(y[keep])
                samples[group][2].extend(weight[keep])
        else:
            ranks = np.arange(1, len(order))
            groups = rank_group(ranks, head, rank_bins)
            x = base[order[0]] - base[order[1:]]
            y = row.reference[order[0]] - row.reference[order[1:]]
            weight = np.maximum(mass[1:], 1e-12)
            for group in range(rank_bins):
                keep = groups == group
                samples[group][0].extend(x[keep])
                samples[group][1].extend(y[keep])
                samples[group][2].extend(weight[keep])
    return [tuple(np.asarray(part, dtype=np.float32) for part in group)
            for group in samples]


def fit_gap_model(rows: list[PairedRow], selected: np.ndarray, scale: float,
                  bias: np.ndarray, temperature: float, head: int, knots: int,
                  rank_bins: int, pairwise: bool) -> GapModel:
    samples = training_samples(rows, selected, scale, bias, temperature, head,
                               rank_bins, pairwise)
    return GapModel(head, rank_bins, [fit_monotonic(*group, knots) for group in samples])


def correct_logits(row: PairedRow, scale: float, bias: np.ndarray, model: GapModel,
                   strength: float, pairwise: bool) -> np.ndarray:
    base = (scale * row.candidate + bias[row.token_ids]).astype(np.float32)
    order = top_order(base, row.exact, model.head)
    if len(order) < 2:
        return base
    calibrated = base[order].copy()
    if pairwise:
        upper, lower = np.triu_indices(len(order), 1)
        gaps = base[order[upper]] - base[order[lower]]
        groups = rank_group(lower, model.head, model.rank_bins)
        mapped = np.empty_like(gaps)
        for group, mapping in enumerate(model.mappings):
            keep = groups == group
            mapped[keep] = predict(mapping, gaps[keep])
        differences = np.zeros((len(order), len(order)), dtype=np.float32)
        differences[upper, lower] = mapped
        differences[lower, upper] = -mapped
        reconstructed = differences.mean(axis=1)
        reconstructed += float(base[order].mean() - reconstructed.mean())
        calibrated = reconstructed
    else:
        ranks = np.arange(1, len(order))
        gaps = base[order[0]] - base[order[1:]]
        groups = rank_group(ranks, model.head, model.rank_bins)
        mapped = np.empty_like(gaps)
        for group, mapping in enumerate(model.mappings):
            keep = groups == group
            mapped[keep] = predict(mapping, gaps[keep])
        calibrated[1:] = calibrated[0] - mapped
    corrected = base.copy()
    corrected[order] += strength * (calibrated - base[order])
    return corrected


def distribution_metrics(row: PairedRow, corrected: np.ndarray,
                         temperature: float) -> tuple[float, float, float, float, float]:
    p = softmax(row.reference / temperature).astype(np.float64)
    q = softmax(corrected / temperature).astype(np.float64)
    log_p, log_q = np.log(p), np.log(q)
    kl = float(np.dot(p, log_p - log_q))
    midpoint = 0.5 * (p + q)
    js = float(0.5 * np.dot(p, log_p - np.log(midpoint))
               + 0.5 * np.dot(q, log_q - np.log(midpoint)))
    tv = float(0.5 * np.abs(p - q).sum())
    top = min(10, len(row.reference))
    ref_top = np.argpartition(row.reference, -top)[-top:]
    candidate_top = np.argpartition(corrected, -top)[-top:]
    top1 = float(np.argmax(row.reference) == np.argmax(corrected))
    top10 = float(np.intersect1d(ref_top, candidate_top, assume_unique=True).size / top)
    return kl, js, tv, top1, top10


def evaluate(rows: list[PairedRow], selected: np.ndarray, temperature: float,
             scale: float, bias: np.ndarray, model: GapModel | None = None,
             strength: float = 0.0, pairwise: bool = False) -> dict[str, float]:
    values = []
    for index in np.flatnonzero(selected):
        row = rows[index]
        corrected = (scale * row.candidate + bias[row.token_ids]) if model is None else \
            correct_logits(row, scale, bias, model, strength, pairwise)
        values.append(distribution_metrics(row, corrected, temperature))
    array = np.asarray(values)
    return dict(zip(("kl", "js", "tv", "top1_agreement", "top10_overlap"), array.mean(0)))


def select_configs(rows: list[PairedRow], inner_train: np.ndarray,
                   inner_validation: np.ndarray, n_vocab: int, temperature: float,
                   heads: list[int], knots_grid: list[int], rank_bins_grid: list[int],
                   strengths: list[float]) -> dict[bool, Config]:
    scale = fit_temperature(rows, inner_train, temperature)
    bias = fit_token_bias(rows, inner_train, n_vocab, scale, temperature)
    baseline = evaluate(rows, inner_validation, temperature, scale, bias)["kl"]
    best: dict[bool, tuple[float, Config]] = {}
    for pairwise in (False, True):
        for head in heads:
            for rank_bins in rank_bins_grid:
                if rank_bins >= head:
                    continue
                for knots in knots_grid:
                    model = fit_gap_model(rows, inner_train, scale, bias, temperature,
                                          head, knots, rank_bins, pairwise)
                    for strength in strengths:
                        value = evaluate(rows, inner_validation, temperature, scale, bias,
                                         model, strength, pairwise)["kl"]
                        config = Config(head, knots, rank_bins, strength)
                        candidate = (value, config)
                        if pairwise not in best or candidate[0] < best[pairwise][0]:
                            best[pairwise] = candidate
        value, config = best[pairwise]
        best[pairwise] = (value, Config(config.head, config.knots, config.rank_bins,
                                        config.strength, value < baseline))
    return {pairwise: value[1] for pairwise, value in best.items()}


def analyze(reference_path: Path, candidate_path: Path, name: str, temperature: float,
            folds: int, heads: list[int], knots_grid: list[int],
            rank_bins_grid: list[int], strengths: list[float]) -> list[dict]:
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
        configs = select_configs(rows, inner_train, inner_validation, n_vocab, temperature,
                                 heads, knots_grid, rank_bins_grid, strengths)

        scale = fit_temperature(rows, train, temperature)
        bias = fit_token_bias(rows, train, n_vocab, scale, temperature)
        methods: dict[str, tuple[dict[str, float], Config | None]] = {
            "identity": (evaluate(rows, test, temperature, 1.0, zero_bias), None),
            "target-temperature": (evaluate(rows, test, temperature, scale, zero_bias), None),
            "target-temperature+token": (evaluate(rows, test, temperature, scale, bias), None),
        }
        baseline = methods["target-temperature+token"][0]
        for pairwise, stem in ((False, "direct-gap"), (True, "pairwise-gap")):
            config = configs[pairwise]
            model = fit_gap_model(rows, train, scale, bias, temperature, config.head,
                                  config.knots, config.rank_bins, pairwise)
            forced = evaluate(rows, test, temperature, scale, bias, model,
                              config.strength, pairwise)
            methods[f"{stem}-forced"] = (forced, config)
            methods[f"{stem}-selected"] = ((forced if config.enabled else baseline), config)
        for method, (metrics, config) in methods.items():
            output.append({
                "quant": name, "fold": fold,
                "test_chunks": "+".join(map(str, test_chunks)),
                "inner_validation_chunks": "+".join(map(str, inner_validation_chunks)),
                "train_positions": int(train.sum()), "test_positions": int(test.sum()),
                "temperature": temperature, "method": method,
                "selected_head": config.head if config else 0,
                "selected_knots": config.knots if config else 0,
                "selected_rank_bins": config.rank_bins if config else 0,
                "selected_strength": config.strength if config else 0.0,
                "selected_enabled": config.enabled if config else False,
                "scale": scale, **metrics,
            })
        print(f"finished {name} fold {fold}: direct={configs[False]}, "
              f"pairwise={configs[True]}", flush=True)
    return output


def write_report(path: Path, rows: list[dict], reference_label: str) -> None:
    identity = np.mean([row["kl"] for row in rows if row["method"] == "identity"])
    methods = list(dict.fromkeys(row["method"] for row in rows))
    lines = [
        "# Monotonic rank-conditioned gap calibration", "",
        f"The target is {reference_label}. Both transforms start from the fitted scalar",
        "temperature plus static token bias. The direct transform maps gaps from the",
        "candidate's top token to lower-ranked head tokens. The pairwise transform maps all",
        "head-token gaps, then reconstructs a single consistent logit vector. Each mapping is",
        "a weighted monotonic piecewise-linear fit. Head size, knot count, rank-bin count, and",
        "correction strength are selected on an inner chunk split. Selected variants can",
        "reject the transform; forced variants retain the best non-null inner configuration.",
        "All metrics come from untouched outer chunks.", "",
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
    lines += ["", "## Inner-selected configurations", "",
              "| Fold | Method | Head | Knots | Rank bins | Strength | Enabled |",
              "|---:|---|---:|---:|---:|---:|:---:|"]
    for row in rows:
        if row["method"] in ("direct-gap-selected", "pairwise-gap-selected"):
            lines.append(
                f"| {row['fold']} | {row['method']} | {row['selected_head']} | "
                f"{row['selected_knots']} | {row['selected_rank_bins']} | "
                f"{row['selected_strength']:g} | "
                f"{'yes' if row['selected_enabled'] else 'no'} |")
    lines += ["", "## Paired outer-block improvements over temperature+token", "",
              "Positive values mean lower KL than the static-bias baseline.", "",
              "| Method | Mean KL reduction | 95% interval | Improved blocks |",
              "|---|---:|---:|---:|"]
    baseline = {int(row["fold"]): row["kl"] for row in rows
                if row["method"] == "target-temperature+token"}
    for method in ("direct-gap-selected", "pairwise-gap-selected"):
        selected = [row for row in rows if row["method"] == method]
        differences = np.array([
            baseline[int(row["fold"])] - row["kl"] for row in selected])
        half_width = 1.96 * differences.std(ddof=1) / np.sqrt(len(differences))
        lines.append(
            f"| {method} | {differences.mean():.7f} | "
            f"[{differences.mean() - half_width:.7f}, "
            f"{differences.mean() + half_width:.7f}] | "
            f"{np.count_nonzero(differences > 0)}/{len(differences)} |")
    lines += ["", "Fold-level values are in `gap_calibration.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--heads", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--knots", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--rank-bins", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(args.reference, args.candidate, args.candidate_name, args.temperature,
                   args.folds, args.heads, args.knots, args.rank_bins, args.strengths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gap_calibration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_dir / "GAP_CALIBRATION.md", rows, args.reference_label)
    print(f"wrote {args.output_dir / 'GAP_CALIBRATION.md'}")


if __name__ == "__main__":
    main()
