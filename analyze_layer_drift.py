#!/usr/bin/env python3
"""Localize BF16/quant residual-stream drift across transformer layers."""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

import numpy as np

from analyze_sparse import load_sparse, softmax


def read_layer_states(path: Path) -> tuple[np.memmap, np.ndarray, np.ndarray, dict[str, int]]:
    with path.open("rb") as stream:
        if stream.read(8) != b"_layers_":
            raise ValueError(f"{path} is not a layer-state capture")
        values = stream.read(28)
        if len(values) != 28:
            raise ValueError(f"truncated layer-state header in {path}")
        version, n_ctx, n_embd, n_chunks, first, rows_per_chunk, n_layers = struct.unpack(
            "<7I", values)
        layers = np.fromfile(stream, dtype="<u4", count=n_layers)
        tokens = np.fromfile(stream, dtype="<i4", count=n_ctx * n_chunks)
    if version != 1:
        raise ValueError(f"unsupported layer-state version {version}")
    if len(layers) != n_layers or len(tokens) != n_ctx * n_chunks:
        raise ValueError(f"truncated layer or token header in {path}")
    if first != n_ctx // 2 or rows_per_chunk != n_ctx - first - 1:
        raise ValueError(f"unexpected evaluated row range in {path}")
    offset = 36 + n_layers * 4 + n_ctx * n_chunks * 4
    shape = (n_chunks, n_layers, rows_per_chunk, n_embd)
    expected = offset + int(np.prod(shape)) * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path} has {path.stat().st_size} bytes; expected {expected}")
    states = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=shape)
    metadata = {"n_ctx": n_ctx, "n_embd": n_embd, "n_chunks": n_chunks,
                "first": first, "rows_per_chunk": rows_per_chunk,
                "n_layers": n_layers}
    return states, layers, tokens, metadata


def check_alignment(reference_layers: np.ndarray, candidate_layers: np.ndarray,
                    reference_tokens: np.ndarray, candidate_tokens: np.ndarray,
                    reference_metadata: dict[str, int],
                    candidate_metadata: dict[str, int]) -> None:
    if reference_metadata != candidate_metadata:
        raise ValueError("reference and candidate layer-state dimensions differ")
    if not np.array_equal(reference_layers, candidate_layers):
        raise ValueError("reference and candidate layer selections differ")
    if not np.array_equal(reference_tokens, candidate_tokens):
        raise ValueError("reference and candidate layer-state tokens differ")


def output_kl(reference_path: Path, candidate_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows, chunks, _, _ = load_sparse(reference_path, candidate_path)
    values = []
    for row in rows:
        p = softmax(row.reference)
        q = softmax(row.candidate)
        values.append(float(np.dot(p, np.log(p) - np.log(q))))
    return np.asarray(values, dtype=np.float64), chunks


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def layer_metrics(reference: np.ndarray, candidate: np.ndarray,
                  kl: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, reference.shape[-1])
    quant = np.asarray(candidate, dtype=np.float64).reshape(-1, candidate.shape[-1])
    delta = ref - quant
    reference_mse = float(np.mean(ref * ref))
    candidate_mse = float(np.mean(quant * quant))
    error_mse = float(np.mean(delta * delta))
    dot = float(np.sum(ref * quant))
    scale = dot / max(float(np.sum(quant * quant)), 1e-30)
    scaled_error_mse = float(np.mean((ref - scale * quant) ** 2))
    row_dot = np.sum(ref * quant, axis=1)
    row_norm = np.sqrt(np.sum(ref * ref, axis=1) * np.sum(quant * quant, axis=1))
    cosine = np.divide(row_dot, row_norm, out=np.zeros_like(row_dot), where=row_norm > 0)
    row_relative_error = np.sqrt(np.mean(delta * delta, axis=1)) / np.maximum(
        np.sqrt(np.mean(ref * ref, axis=1)), 1e-12)
    mean_delta = delta.mean(axis=0)
    return {
        "reference_rms": float(np.sqrt(reference_mse)),
        "candidate_rms": float(np.sqrt(candidate_mse)),
        "error_rms": float(np.sqrt(error_mse)),
        "relative_error": float(np.sqrt(error_mse / max(reference_mse, 1e-30))),
        "mean_cosine": float(cosine.mean()),
        "best_candidate_scale": scale,
        "scaled_relative_error": float(
            np.sqrt(scaled_error_mse / max(reference_mse, 1e-30))),
        "mean_delta_error_fraction": float(
            np.mean(mean_delta * mean_delta) / max(error_mse, 1e-30)),
        "row_error_kl_correlation": safe_correlation(row_relative_error, kl),
    }


def heldout_bias_metrics(reference: np.ndarray, candidate: np.ndarray,
                         folds: int) -> list[dict[str, float]]:
    n_chunks = reference.shape[0]
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    output = []
    for fold in range(folds):
        test_chunks = np.arange(fold * fold_size, (fold + 1) * fold_size)
        train_chunks = np.array([chunk for chunk in range(n_chunks)
                                 if chunk not in test_chunks])
        train_delta = (np.asarray(reference[train_chunks], dtype=np.float64)
                       - np.asarray(candidate[train_chunks], dtype=np.float64))
        bias = train_delta.mean(axis=(0, 1))
        test_reference = np.asarray(reference[test_chunks], dtype=np.float64)
        test_candidate = np.asarray(candidate[test_chunks], dtype=np.float64)
        raw_mse = float(np.mean((test_reference - test_candidate) ** 2))
        corrected_mse = float(np.mean((test_reference - test_candidate - bias) ** 2))
        output.append({
            "fold": fold, "test_chunks": "+".join(map(str, test_chunks)),
            "raw_mse": raw_mse, "corrected_mse": corrected_mse,
            "recovered": 1.0 - corrected_mse / raw_mse,
            "bias_rms": float(np.sqrt(np.mean(bias * bias))),
        })
    return output


def analyze(reference_states: np.memmap, candidate_states: np.memmap,
            layers: np.ndarray, kl: np.ndarray, folds: int) -> tuple[list[dict], list[dict]]:
    summary, heldout = [], []
    expected_rows = reference_states.shape[0] * reference_states.shape[2]
    if len(kl) != expected_rows:
        raise ValueError(f"{len(kl)} output rows do not align with {expected_rows} layer rows")
    for layer_index, layer in enumerate(layers):
        reference = reference_states[:, layer_index]
        candidate = candidate_states[:, layer_index]
        fold_rows = heldout_bias_metrics(reference, candidate, folds)
        for row in fold_rows:
            heldout.append({"layer": int(layer), **row})
        metrics = layer_metrics(reference, candidate, kl)
        metrics["heldout_bias_recovery"] = float(np.mean(
            [row["recovered"] for row in fold_rows]))
        summary.append({"layer": int(layer), **metrics})
        print(f"finished layer {int(layer)}: relative error={metrics['relative_error']:.2%}, "
              f"cosine={metrics['mean_cosine']:.6f}", flush=True)
    return summary, heldout


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: list[dict], reference_label: str,
                 candidate_label: str) -> None:
    lines = [
        "# Residual-stream quantization drift by layer",
        "",
        f"Aligned teacher-forced states from {reference_label} and {candidate_label} are compared",
        "at the input of every transformer block. Metrics cover the same evaluated positions as",
        "the saved output distributions. Relative error is state-error RMS divided by reference",
        "state RMS. The scale-adjusted column removes one globally fitted scalar only as a",
        "diagnostic. Static-bias recovery uses four held-out blocks of complete context chunks.",
        "",
        "Qwen3.5 uses full attention in layers 3, 7, ..., 31; the other layers are recurrent",
        "gated-delta blocks. A row at layer L measures all drift accumulated before block L.",
        "",
        "| Layer | Type | Ref RMS | Error RMS | Relative | Scaled relative | Cosine | Static-bias recovery | Error/KL corr. |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        layer = int(row["layer"])
        layer_type = "full attention" if layer % 4 == 3 else "recurrent"
        lines.append(
            f"| {layer} | {layer_type} | {row['reference_rms']:.5f} | "
            f"{row['error_rms']:.5f} | {row['relative_error']:.2%} | "
            f"{row['scaled_relative_error']:.2%} | {row['mean_cosine']:.6f} | "
            f"{row['heldout_bias_recovery']:.2%} | "
            f"{row['row_error_kl_correlation']:.3f} |")
    largest_jump = max(zip(summary[:-1], summary[1:]),
                       key=lambda pair: pair[1]["relative_error"] - pair[0]["relative_error"])
    jump = largest_jump[1]["relative_error"] - largest_jump[0]["relative_error"]
    peak = max(summary, key=lambda row: row["relative_error"])
    first_full = next(row for row in summary if int(row["layer"]) == 3)
    last = summary[-1]
    lines += [
        "",
        f"The largest adjacent increase in relative error is from layer "
        f"{largest_jump[0]['layer']} to {largest_jump[1]['layer']} ({jump:+.2%}). The input",
        f"embedding starts at {summary[0]['relative_error']:.2%} error, but drift reaches",
        f"{first_full['relative_error']:.2%} before the first full-attention block. Relative error",
        f"peaks at layer {peak['layer']} ({peak['relative_error']:.2%}); by layer {last['layer']},",
        f"its per-row correlation with final KL is {last['row_error_kl_correlation']:.3f}.",
        "",
        "Machine-readable layer summaries are in `layer_drift.csv`; held-out static-bias folds",
        "are in `layer_bias_folds.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-states", type=Path, required=True)
    parser.add_argument("--candidate-states", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--reference-label", default="BF16")
    parser.add_argument("--candidate-label", default="Q2_K_XL")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/bf16_layer_drift_q2"))
    args = parser.parse_args()

    reference_states, reference_layers, reference_tokens, reference_metadata = read_layer_states(
        args.reference_states)
    candidate_states, candidate_layers, candidate_tokens, candidate_metadata = read_layer_states(
        args.candidate_states)
    check_alignment(reference_layers, candidate_layers, reference_tokens, candidate_tokens,
                    reference_metadata, candidate_metadata)
    kl, chunks = output_kl(args.reference_logits, args.candidate_logits)
    expected_chunks = np.repeat(
        np.arange(reference_metadata["n_chunks"]), reference_metadata["rows_per_chunk"])
    if not np.array_equal(chunks, expected_chunks):
        raise ValueError("logit and layer-state chunk order differs")
    summary, heldout = analyze(
        reference_states, candidate_states, reference_layers, kl, args.folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "layer_drift.csv", summary)
    write_csv(args.output_dir / "layer_bias_folds.csv", heldout)
    write_report(args.output_dir / "LAYER_DRIFT.md", summary,
                 args.reference_label, args.candidate_label)
    print(f"wrote {args.output_dir / 'LAYER_DRIFT.md'}")


if __name__ == "__main__":
    main()
