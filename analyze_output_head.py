#!/usr/bin/env python3
"""Evaluate a sparse high-precision output-head sidecar for a quantized model."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from analyze_hidden_adapter import load_hidden
from analyze_sparse import PairedRow, load_sparse, softmax


def load_weight_rows(model_path: Path, token_ids: np.ndarray,
                     gguf_python_path: Path) -> tuple[np.ndarray, str]:
    sys.path.insert(0, str(gguf_python_path))
    try:
        from gguf import GGUFReader, dequantize
    except ImportError as error:
        raise RuntimeError(
            f"could not import gguf from {gguf_python_path}; pass llama.cpp/gguf-py") from error

    reader = GGUFReader(model_path, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    tensor = tensors.get("output.weight", tensors.get("token_embd.weight"))
    if tensor is None:
        raise ValueError(f"{model_path} has neither output.weight nor token_embd.weight")
    if len(tensor.shape) != 2 or int(tensor.shape[1]) <= int(token_ids.max()):
        raise ValueError(f"output tensor shape {tensor.shape} does not cover requested tokens")
    weights = dequantize(tensor.data[token_ids], tensor.tensor_type)
    return np.asarray(weights, dtype=np.float32), tensor.tensor_type.name


def top_indices(rows: list[PairedRow], head: int) -> tuple[list[np.ndarray], np.ndarray]:
    per_row = []
    for row in rows:
        count = min(head, len(row.candidate))
        indices = np.argpartition(row.candidate, -count)[-count:]
        indices = indices[np.argsort(row.candidate[indices])[::-1]]
        per_row.append(indices)
    tokens = np.unique(np.concatenate([row.token_ids[index]
                                       for row, index in zip(rows, per_row)]))
    return per_row, tokens.astype(np.int32)


def compute_corrections(rows: list[PairedRow], hidden: np.ndarray,
                        per_row_top: list[np.ndarray], unique_tokens: np.ndarray,
                        weight_delta: np.ndarray) -> list[np.ndarray]:
    corrections = []
    for row_index, (row, local_top) in enumerate(zip(rows, per_row_top)):
        positions = np.searchsorted(unique_tokens, row.token_ids[local_top])
        if not np.array_equal(unique_tokens[positions], row.token_ids[local_top]):
            raise ValueError("top token is absent from output-weight sidecar")
        corrections.append(np.asarray(
            weight_delta[positions] @ np.asarray(hidden[row_index], dtype=np.float32),
            dtype=np.float32))
    return corrections


def evaluate(rows: list[PairedRow], selected: np.ndarray, per_row_top: list[np.ndarray],
             corrections: list[np.ndarray], head: int, strength: float) -> dict[str, float]:
    kl, js, tv, top1, top10 = [], [], [], [], []
    for row_index in np.flatnonzero(selected):
        row = rows[row_index]
        p = softmax(row.reference)
        corrected = row.candidate.copy()
        local = per_row_top[row_index][:head]
        corrected[local] += strength * corrections[row_index][:head]
        q = softmax(corrected)
        log_p = np.log(p)
        log_q = np.log(q)
        kl.append(float(np.dot(p, log_p - log_q)))
        midpoint = 0.5 * (p + q)
        js.append(float(0.5 * np.dot(p, log_p - np.log(midpoint))
                        + 0.5 * np.dot(q, log_q - np.log(midpoint))))
        tv.append(float(0.5 * np.abs(p - q).sum()))
        top_count = min(10, len(row.reference))
        ref_top = np.argpartition(row.reference, -top_count)[-top_count:]
        candidate_top = np.argpartition(corrected, -top_count)[-top_count:]
        top1.append(int(np.argmax(row.reference) == np.argmax(corrected)))
        top10.append(
            np.intersect1d(ref_top, candidate_top, assume_unique=True).size / top_count)
    return {
        "kl": float(np.mean(kl)), "js": float(np.mean(js)), "tv": float(np.mean(tv)),
        "top1_agreement": float(np.mean(top1)), "top10_overlap": float(np.mean(top10)),
    }


def centered_rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = (left - left.mean()) - (right - right.mean())
    return float(np.sqrt(np.mean(difference.astype(np.float64) ** 2)))


def diagnostics(rows: list[PairedRow], hidden: np.ndarray, per_row_top: list[np.ndarray],
                unique_tokens: np.ndarray, quant_weights: np.ndarray,
                corrections: list[np.ndarray]) -> dict[str, float]:
    quant_error, correction_values, residual_values = [], [], []
    for row_index, (row, local) in enumerate(zip(rows, per_row_top)):
        positions = np.searchsorted(unique_tokens, row.token_ids[local])
        recomputed = quant_weights[positions] @ np.asarray(hidden[row_index], dtype=np.float32)
        quant_error.append(centered_rmse(recomputed, row.candidate[local]))
        correction = corrections[row_index]
        residual = row.reference[local] - row.candidate[local]
        correction_values.append(correction - correction.mean())
        residual_values.append(residual - residual.mean())
    correction_flat = np.concatenate(correction_values).astype(np.float64)
    residual_flat = np.concatenate(residual_values).astype(np.float64)
    return {
        "quant_recompute_centered_rmse": float(np.mean(quant_error)),
        "head_correction_rms": float(np.sqrt(np.mean(correction_flat ** 2))),
        "target_residual_rms": float(np.sqrt(np.mean(residual_flat ** 2))),
        "head_target_correlation": float(np.corrcoef(correction_flat, residual_flat)[0, 1]),
    }


def analyze(rows: list[PairedRow], chunks: np.ndarray, n_chunks: int,
            per_row_top: list[np.ndarray], corrections: list[np.ndarray], folds: int,
            heads: list[int], strengths: list[float]) -> tuple[list[dict], list[dict]]:
    if n_chunks % folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {folds} folds")
    fold_size = n_chunks // folds
    curves, selected_rows = [], []
    for fold in range(folds):
        test_chunks = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_chunks = [chunk for chunk in range(n_chunks) if chunk not in test_chunks]
        validation_chunks = train_chunks[fold % 4::4]
        validation = np.isin(chunks, validation_chunks)
        test = np.isin(chunks, test_chunks)
        options = [(evaluate(rows, validation, per_row_top, corrections, head, strength)["kl"],
                    head, strength) for head in heads for strength in strengths]
        selected_kl, selected_head, selected_strength = min(options)
        baseline = evaluate(rows, test, per_row_top, corrections, 0, 0.0)
        result = evaluate(rows, test, per_row_top, corrections,
                          selected_head, selected_strength)
        selected_rows.extend([
            {"fold": fold, "test_chunks": "+".join(map(str, test_chunks)),
             "validation_chunks": "+".join(map(str, validation_chunks)),
             "method": "identity", "selected_head": selected_head,
             "selected_strength": selected_strength, "validation_kl": selected_kl,
             **baseline},
            {"fold": fold, "test_chunks": "+".join(map(str, test_chunks)),
             "validation_chunks": "+".join(map(str, validation_chunks)),
             "method": "output-head-sidecar", "selected_head": selected_head,
             "selected_strength": selected_strength, "validation_kl": selected_kl,
             **result},
        ])
        for head in heads:
            for strength in strengths:
                curves.append({
                    "fold": fold, "test_chunks": "+".join(map(str, test_chunks)),
                    "head": head, "strength": strength,
                    **evaluate(rows, test, per_row_top, corrections, head, strength),
                })
        print(f"finished fold {fold}: selected head={selected_head}, "
              f"strength={selected_strength:g}", flush=True)
    return curves, selected_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: list[dict], method: str, metric: str) -> float:
    return float(np.mean([row[metric] for row in rows if row["method"] == method]))


def write_report(path: Path, selected: list[dict], curves: list[dict], diagnostics_: dict,
                 reference_type: str, quant_type: str, unique_tokens: int,
                 dimensions: int, full_sidecar_bytes: int) -> None:
    raw = mean_metric(selected, "identity", "kl")
    sidecar = mean_metric(selected, "output-head-sidecar", "kl")
    chosen = [(row["selected_head"], row["selected_strength"])
              for row in selected if row["method"] == "output-head-sidecar"]
    raw_by_fold = {int(row["fold"]): row["kl"] for row in selected
                   if row["method"] == "identity"}
    sidecar_by_fold = {int(row["fold"]): row["kl"] for row in selected
                       if row["method"] == "output-head-sidecar"}
    reductions = np.array([raw_by_fold[fold] - sidecar_by_fold[fold]
                           for fold in sorted(raw_by_fold)], dtype=np.float64)
    reduction_mean = float(reductions.mean())
    # Student-t critical value for the standard four-fold experiment (three df).
    half_width = float(3.182 * reductions.std(ddof=1) / np.sqrt(len(reductions)))
    full_mib = full_sidecar_bytes / 2**20
    lines = [
        "# Sparse high-precision output-head sidecar",
        "",
        f"The quantized tied embedding/output tensor is `{quant_type}` and the reference is",
        f"`{reference_type}`. For each row, this experiment adds `(W_ref - W_quant) h` only",
        "to the quant candidate's top-N tokens. N and a blend strength are selected on a chunk",
        "split inside each outer fold. The reference logits in held-out chunks are used only for",
        "evaluation and hyperparameter selection, never to construct the weight correction.",
        "",
        f"The sweep loaded {unique_tokens:,} vocabulary rows of {dimensions:,} values. A full",
        f"BF16 output sidecar would occupy about {full_mib:.1f} MiB before metadata.",
        "",
        "## Selected held-out result",
        "",
        "| Method | KL | Recovered | JS | TV | Top-1 | Top-10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in (("identity", "Raw quant"),
                          ("output-head-sidecar", "Selected head sidecar")):
        kl = mean_metric(selected, method, "kl")
        lines.append(
            f"| {label} | {kl:.7f} | {1.0 - kl / raw:.2%} | "
            f"{mean_metric(selected, method, 'js'):.7f} | "
            f"{mean_metric(selected, method, 'tv'):.7f} | "
            f"{mean_metric(selected, method, 'top1_agreement'):.1%} | "
            f"{mean_metric(selected, method, 'top10_overlap'):.1%} |")
    lines += [
        "",
        f"Fold selections (head, strength): {', '.join(f'({h}, {s:g})' for h, s in chosen)}.",
        f"The mean per-fold KL reduction is {reduction_mean:.7f}; its four-block t interval is",
        f"[{reduction_mean - half_width:.7f}, {reduction_mean + half_width:.7f}].",
        "",
        "## Alignment diagnostics",
        "",
        f"- Recomputed quant-logit centered RMSE: {diagnostics_['quant_recompute_centered_rmse']:.6f}",
        f"- Output-head correction RMS: {diagnostics_['head_correction_rms']:.6f}",
        f"- Full BF16-vs-quant residual RMS on the same tokens: {diagnostics_['target_residual_rms']:.6f}",
        f"- Correction/residual correlation: {diagnostics_['head_target_correlation']:.4f}",
        "",
        "The first diagnostic verifies that the saved final hidden state feeds the inspected tied",
        "output tensor. Fold-level selections are in `output_head_selected.csv`; the complete",
        "held-out sweep is in `output_head_curves.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--gguf-python-path", type=Path,
                        default=Path("/home/eapache/src/llama.cpp/gguf-py"))
    parser.add_argument("--heads", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/bf16_output_head_q2"))
    args = parser.parse_args()

    if min(args.heads) <= 0 or max(args.heads) > 4096:
        parser.error("heads must be in [1, 4096]")
    if any(strength < 0 for strength in args.strengths):
        parser.error("strengths must be nonnegative")
    rows, chunks, n_vocab, n_chunks = load_sparse(args.reference, args.candidate)
    hidden = load_hidden(args.hidden, args.candidate)
    per_row_top, unique_tokens = top_indices(rows, max(args.heads))
    reference_weights, reference_type = load_weight_rows(
        args.reference_model, unique_tokens, args.gguf_python_path)
    quant_weights, quant_type = load_weight_rows(
        args.candidate_model, unique_tokens, args.gguf_python_path)
    if reference_weights.shape != quant_weights.shape or reference_weights.shape[1] != hidden.shape[1]:
        raise ValueError("reference, candidate, and hidden dimensions do not align")
    weight_delta = reference_weights - quant_weights
    corrections = compute_corrections(
        rows, hidden, per_row_top, unique_tokens, weight_delta)
    diagnostics_ = diagnostics(rows, hidden, per_row_top, unique_tokens,
                               quant_weights, corrections)
    curves, selected = analyze(rows, chunks, n_chunks, per_row_top, corrections,
                               args.folds, args.heads, args.strengths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "output_head_curves.csv", curves)
    write_csv(args.output_dir / "output_head_selected.csv", selected)
    write_report(args.output_dir / "OUTPUT_HEAD.md", selected, curves, diagnostics_,
                 reference_type, quant_type, len(unique_tokens), hidden.shape[1],
                 n_vocab * hidden.shape[1] * 2)
    print(f"wrote {args.output_dir / 'OUTPUT_HEAD.md'}")


if __name__ == "__main__":
    main()
