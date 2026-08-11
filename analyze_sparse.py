#!/usr/bin/env python3
"""Memory-efficient calibration curves for larger llama.cpp KLD captures."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_real_logits import check_alignment, read_kld


@dataclass
class PairedRow:
    token_ids: np.ndarray
    reference: np.ndarray
    candidate: np.ndarray
    exact: np.ndarray


def softmax(values: np.ndarray) -> np.ndarray:
    weights = np.exp(values - values.max(), dtype=np.float32)
    return weights / weights.sum(dtype=np.float64)


def load_sparse(reference_path: Path, candidate_path: Path) -> tuple[list[PairedRow], np.ndarray, int, int]:
    reference = read_kld(reference_path)
    candidate = read_kld(candidate_path)
    check_alignment(reference, candidate)
    rows = []
    for row in range(reference.n_rows):
        ref_codes = reference.codes[row, : reference.n_vocab]
        candidate_codes = candidate.codes[row, : candidate.n_vocab]
        active = (ref_codes > 0) | (candidate_codes > 0)
        token_ids = np.flatnonzero(active).astype(np.int32)
        ref_values = reference.minimum[row] + reference.scale[row] * ref_codes[token_ids]
        candidate_values = candidate.minimum[row] + candidate.scale[row] * candidate_codes[token_ids]
        rows.append(PairedRow(
            token_ids,
            ref_values.astype(np.float32),
            candidate_values.astype(np.float32),
            ((ref_codes[token_ids] > 0) & (candidate_codes[token_ids] > 0)),
        ))
    return rows, reference.chunk_ids, reference.n_vocab, reference.n_chunk


def fit_temperature(rows: list[PairedRow], selected: np.ndarray, temperature: float) -> float:
    selected_rows = [rows[index] for index in np.flatnonzero(selected)]
    target = 0.0
    for row in selected_rows:
        p = softmax(row.reference / temperature)
        target += float(np.dot(p, row.candidate))
    target /= len(selected_rows)
    scale = 1.0
    for _ in range(8):
        expectation = 0.0
        variance = 0.0
        for row in selected_rows:
            q = softmax(scale * row.candidate / temperature)
            mean = float(np.dot(q, row.candidate))
            expectation += mean
            variance += float(np.dot(q, (row.candidate - mean) ** 2))
        expectation /= len(selected_rows)
        variance /= len(selected_rows)
        step = ((expectation - target) / temperature) / max(
            variance / (temperature * temperature), 1e-12)
        scale = float(np.clip(scale - step, 0.4, 1.8))
        if abs(step) < 1e-6:
            break
    return scale


def fit_token_bias(rows: list[PairedRow], selected: np.ndarray, n_vocab: int,
                   scale: float, temperature: float, prior_positions: float = 32.0) -> np.ndarray:
    token_parts = []
    residual_parts = []
    for index in np.flatnonzero(selected):
        row = rows[index]
        p = softmax(row.reference / temperature)
        residual = row.reference - scale * row.candidate
        residual -= float(np.dot(p, residual))
        token_parts.append(row.token_ids[row.exact])
        residual_parts.append(residual[row.exact])
    tokens = np.concatenate(token_parts)
    residuals = np.concatenate(residual_parts)
    counts = np.bincount(tokens, minlength=n_vocab)
    numerator = np.bincount(tokens, weights=residuals, minlength=n_vocab)
    bias = numerator / np.maximum(counts + prior_positions, 1.0)
    observed = counts > 0
    bias -= np.average(bias[observed], weights=counts[observed])
    return np.clip(bias, -1.0, 1.0).astype(np.float32)


def candidate_entropy(row: PairedRow, scale: float, temperature: float) -> float:
    q = softmax(scale * row.candidate / temperature)
    return float(-np.dot(q, np.log(q)))


def fit_entropy_bias(rows: list[PairedRow], selected: np.ndarray, n_vocab: int,
                     scale: float, temperature: float) -> tuple[np.ndarray, list[np.ndarray]]:
    indices = np.flatnonzero(selected)
    entropies = np.array([candidate_entropy(rows[index], scale, temperature) for index in indices])
    thresholds = np.quantile(entropies, [1 / 3, 2 / 3])
    groups = np.searchsorted(thresholds, entropies, side="right")
    biases = []
    for group in range(3):
        group_selection = np.zeros(len(rows), dtype=bool)
        group_selection[indices[groups == group]] = True
        biases.append(fit_token_bias(rows, group_selection, n_vocab, scale, temperature))
    return thresholds, biases


def metrics(rows: list[PairedRow], selected: np.ndarray, temperature: float,
            scale: float = 1.0, bias: np.ndarray | None = None,
            entropy_model: tuple[np.ndarray, list[np.ndarray]] | None = None) -> dict[str, float]:
    kl = []
    js = []
    tv = []
    top1 = []
    top10 = []
    for index in np.flatnonzero(selected):
        row = rows[index]
        p = softmax(row.reference / temperature)
        corrected = scale * row.candidate
        if bias is not None:
            corrected = corrected + bias[row.token_ids]
        if entropy_model is not None:
            thresholds, biases = entropy_model
            group = int(np.searchsorted(
                thresholds, candidate_entropy(row, scale, temperature), side="right"))
            corrected = corrected + biases[group][row.token_ids]
        q = softmax(corrected / temperature)
        lp = np.log(p)
        lq = np.log(q)
        kl.append(float(np.dot(p, lp - lq)))
        midpoint = 0.5 * (p + q)
        js.append(float(0.5 * np.dot(p, lp - np.log(midpoint))
                        + 0.5 * np.dot(q, lq - np.log(midpoint))))
        tv.append(float(0.5 * np.abs(p - q).sum()))
        # KLD compression can leave fewer than ten non-clipped tokens in the union.
        # In that case the tail ordering is unobservable, so compare the largest
        # observable head rather than failing or inventing an ordering for clipped ties.
        top_k = min(10, len(row.token_ids))
        ref_top = np.argpartition(row.reference, -top_k)[-top_k:]
        candidate_top = np.argpartition(corrected, -top_k)[-top_k:]
        top1.append(int(np.argmax(row.reference) == np.argmax(corrected)))
        top10.append(
            np.intersect1d(ref_top, candidate_top, assume_unique=True).size / top_k)
    return {
        "kl": float(np.mean(kl)),
        "js": float(np.mean(js)),
        "tv": float(np.mean(tv)),
        "acceptance": float(1.0 - np.mean(tv)),
        "top1_agreement": float(np.mean(top1)),
        "top10_overlap": float(np.mean(top10)),
    }


def analyze_quant(reference_path: Path, candidate_path: Path, name: str,
                  temperatures: list[float], chunk_counts: list[int], folds: int) -> list[dict]:
    rows, chunks, n_vocab, n_chunks = load_sparse(reference_path, candidate_path)
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    if max(chunk_counts) > n_chunks - fold_size:
        raise ValueError("calibration chunk count exceeds chunks outside an outer fold")
    output = []
    for fold in range(folds):
        test_chunks = list(range(fold * fold_size, (fold + 1) * fold_size))
        test = np.isin(chunks, test_chunks)
        order = [(fold * fold_size + fold_size + offset) % n_chunks for offset in range(n_chunks)]
        available = [chunk for chunk in order if chunk not in test_chunks]
        for chunk_count in chunk_counts:
            train_chunks = available[:chunk_count]
            train = np.isin(chunks, train_chunks)
            for temperature in temperatures:
                scale = fit_temperature(rows, train, temperature)
                token_bias = fit_token_bias(rows, train, n_vocab, scale, temperature)
                entropy_bias = fit_entropy_bias(rows, train, n_vocab, scale, temperature)
                methods = {
                    "identity": metrics(rows, test, temperature),
                    "target-temperature": metrics(rows, test, temperature, scale=scale),
                    "target-temperature+token": metrics(
                        rows, test, temperature, scale=scale, bias=token_bias),
                    "target-temperature+entropy-token": metrics(
                        rows, test, temperature, scale=scale, entropy_model=entropy_bias),
                }
                for method, result in methods.items():
                    output.append({
                        "quant": name,
                        "fold": fold,
                        "test_chunks": "+".join(map(str, test_chunks)),
                        "train_chunks": "+".join(map(str, train_chunks)),
                        "calibration_chunks": chunk_count,
                        "calibration_positions": int(train.sum()),
                        "test_positions": int(test.sum()),
                        "temperature": temperature,
                        "method": method,
                        "scale": scale,
                        "equivalent_quant_temperature": temperature / scale,
                        **result,
                    })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], reference_label: str) -> None:
    lines = [
        "# Expanded sparse calibration results",
        "",
        f"The compact analyzer retains only the union of non-clipped {reference_label}/candidate tokens per",
        "position. Four outer folds hold out complete blocks of context chunks. Temperature",
        "is fitted at the downstream temperature. The entropy-token method learns separate",
        "token biases for low-, middle-, and high-entropy calibration tertiles.",
        "",
    ]
    methods = ["identity", "target-temperature", "target-temperature+token",
               "target-temperature+entropy-token"]
    for quant in sorted({row["quant"] for row in rows}):
        for temperature in sorted({float(row["temperature"]) for row in rows}):
            lines += [f"## {quant} at T={temperature:g}", "",
                      "| Calibration positions | Method | KL | Recovered | Top-1 | Scale SD |",
                      "|---:|---|---:|---:|---:|---:|"]
            positions_set = sorted({int(row["calibration_positions"]) for row in rows})
            for positions in positions_set:
                selected = [row for row in rows if row["quant"] == quant
                            and float(row["temperature"]) == temperature
                            and int(row["calibration_positions"]) == positions]
                identity = np.mean([row["kl"] for row in selected if row["method"] == "identity"])
                scales = np.array([row["scale"] for row in selected if row["method"] == "target-temperature"])
                for method in methods:
                    values = [row for row in selected if row["method"] == method]
                    kl = np.mean([row["kl"] for row in values])
                    top1 = np.mean([row["top1_agreement"] for row in values])
                    lines.append(f"| {positions} | {method} | {kl:.7f} | {1.0 - kl / identity:.2%} | {top1:.1%} | {scales.std(ddof=1):.6f} |")
            lines.append("")
    lines += ["Fold-level values are in `sparse_calibration.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--q8", type=Path)
    parser.add_argument("--q4", type=Path)
    parser.add_argument("--q2", type=Path)
    parser.add_argument("--reference-label", default="Q8")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0])
    parser.add_argument("--chunk-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24])
    parser.add_argument("--folds", type=int, default=4)
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
        futures = [executor.submit(analyze_quant, args.reference, path, name,
                                   args.temperatures, args.chunk_counts, args.folds)
                   for name, path in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"finished {result[0]['quant']}", flush=True)
            rows.extend(result)
    rows.sort(key=lambda row: (row["quant"], row["fold"], row["calibration_positions"],
                               row["temperature"], row["method"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sparse_calibration.csv", rows)
    write_report(args.output_dir / "REPORT.md", rows, args.reference_label)
    print(f"wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
