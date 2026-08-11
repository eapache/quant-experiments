#!/usr/bin/env python3
"""Build fold-clean mean residual-stream control vectors as tiny GGUF sidecars."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from analyze_layer_drift import check_alignment, read_layer_states


def mean_vector(reference: np.ndarray, candidate: np.ndarray,
                train_chunks: np.ndarray) -> np.ndarray:
    delta = (np.asarray(reference[train_chunks], dtype=np.float64)
             - np.asarray(candidate[train_chunks], dtype=np.float64))
    return np.asarray(delta.mean(axis=(0, 1)), dtype=np.float32)


def write_control_vector(path: Path, vector: np.ndarray, control_layer: int,
                         gguf_python_path: Path) -> None:
    sys.path.insert(0, str(gguf_python_path))
    try:
        from gguf import GGUFWriter
    except ImportError as error:
        raise RuntimeError(
            f"could not import gguf from {gguf_python_path}; pass llama.cpp/gguf-py") from error
    writer = GGUFWriter(path, "controlvector")
    writer.add_string("controlvector.model_hint", "Qwen3.5-4B BF16-minus-Q2 mean state")
    writer.add_tensor(f"direction.{control_layer}", np.asarray(vector, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-states", type=Path, required=True)
    parser.add_argument("--candidate-states", type=Path, required=True)
    parser.add_argument("--state-layer", type=int, default=3,
                        help="layer input whose reference-minus-candidate mean is fitted")
    parser.add_argument("--control-layer", type=int, default=2,
                        help="block output where llama.cpp adds the vector")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--gguf-python-path", type=Path,
                        default=Path("/home/eapache/src/llama.cpp/gguf-py"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/bf16_mean_control_q2"))
    args = parser.parse_args()

    reference, reference_layers, reference_tokens, reference_metadata = read_layer_states(
        args.reference_states)
    candidate, candidate_layers, candidate_tokens, candidate_metadata = read_layer_states(
        args.candidate_states)
    check_alignment(reference_layers, candidate_layers, reference_tokens, candidate_tokens,
                    reference_metadata, candidate_metadata)
    matches = np.flatnonzero(reference_layers == args.state_layer)
    if len(matches) != 1:
        raise ValueError(f"state layer {args.state_layer} is absent or duplicated")
    if args.control_layer < 1 or args.control_layer >= reference_metadata["n_layers"]:
        raise ValueError("control layer must be in the model's usable range [1, n_layers)")
    n_chunks = reference_metadata["n_chunks"]
    if n_chunks % args.folds:
        raise ValueError(f"{n_chunks} chunks cannot be evenly divided into {args.folds} folds")
    layer_index = int(matches[0])
    reference_layer = reference[:, layer_index]
    candidate_layer = candidate[:, layer_index]
    fold_size = n_chunks // args.folds
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for fold in range(args.folds):
        test_chunks = np.arange(fold * fold_size, (fold + 1) * fold_size)
        train_chunks = np.array([chunk for chunk in range(n_chunks)
                                 if chunk not in test_chunks])
        vector = mean_vector(reference_layer, candidate_layer, train_chunks)
        filename = f"mean_control_layer{args.control_layer}_fold{fold}.gguf"
        write_control_vector(args.output_dir / filename, vector, args.control_layer,
                             args.gguf_python_path)
        manifest.append({
            "fold": fold, "state_layer": args.state_layer,
            "control_layer": args.control_layer,
            "train_chunks": "+".join(map(str, train_chunks)),
            "test_chunks": "+".join(map(str, test_chunks)),
            "dimensions": len(vector),
            "vector_rms": float(np.sqrt(np.mean(vector.astype(np.float64) ** 2))),
            "vector_norm": float(np.linalg.norm(vector.astype(np.float64))),
            "filename": filename,
        })
    all_vector = mean_vector(reference_layer, candidate_layer, np.arange(n_chunks))
    all_filename = f"mean_control_layer{args.control_layer}_all.gguf"
    write_control_vector(args.output_dir / all_filename, all_vector, args.control_layer,
                         args.gguf_python_path)
    manifest.append({
        "fold": "all", "state_layer": args.state_layer,
        "control_layer": args.control_layer,
        "train_chunks": "+".join(map(str, range(n_chunks))), "test_chunks": "",
        "dimensions": len(all_vector),
        "vector_rms": float(np.sqrt(np.mean(all_vector.astype(np.float64) ** 2))),
        "vector_norm": float(np.linalg.norm(all_vector.astype(np.float64))),
        "filename": all_filename,
    })
    with (args.output_dir / "control_vectors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote {len(manifest)} vectors to {args.output_dir}")


if __name__ == "__main__":
    main()
