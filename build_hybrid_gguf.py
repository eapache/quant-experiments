#!/usr/bin/env python3
"""Build a GGUF that replaces selected quantized-model blocks from a donor GGUF."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np


LAYER_NAME = re.compile(r"^blk\.(\d+)\.")


def tensor_layer(name: str) -> int | None:
    match = LAYER_NAME.match(name)
    return int(match.group(1)) if match else None


def use_donor(name: str, layers: set[int]) -> bool:
    layer = tensor_layer(name)
    return layer is not None and layer in layers


def digest(data: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(data))).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True,
                        help="GGUF supplying metadata and all untouched tensors")
    parser.add_argument("--donor", type=Path, required=True,
                        help="shape-compatible GGUF supplying replacement blocks")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf-python-path", type=Path,
                        default=Path("/home/eapache/src/llama.cpp/gguf-py"))
    args = parser.parse_args()
    if args.output.resolve() in (args.base.resolve(), args.donor.resolve()):
        parser.error("output must differ from base and donor")
    layers = set(args.layers)
    if not layers or min(layers) < 0:
        parser.error("layers must be nonnegative")

    sys.path.insert(0, str(args.gguf_python_path))
    try:
        import gguf
    except ImportError as error:
        raise RuntimeError(
            f"could not import gguf from {args.gguf_python_path}; pass llama.cpp/gguf-py") from error

    base = gguf.GGUFReader(args.base, "r")
    donor = gguf.GGUFReader(args.donor, "r")
    base_tensors = {tensor.name: tensor for tensor in base.tensors}
    donor_tensors = {tensor.name: tensor for tensor in donor.tensors}
    if base_tensors.keys() != donor_tensors.keys():
        missing = sorted(base_tensors.keys() - donor_tensors.keys())
        extra = sorted(donor_tensors.keys() - base_tensors.keys())
        raise ValueError(f"tensor name sets differ; missing={missing[:5]}, extra={extra[:5]}")
    for name, tensor in base_tensors.items():
        if tuple(tensor.shape) != tuple(donor_tensors[name].shape):
            raise ValueError(
                f"tensor shape differs for {name}: {tensor.shape} vs {donor_tensors[name].shape}")
    available_layers = {layer for name in base_tensors
                        if (layer := tensor_layer(name)) is not None}
    if not layers <= available_layers:
        raise ValueError(f"requested absent layers: {sorted(layers - available_layers)}")

    arch = base.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=base.endianess)
    alignment_field = base.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()
    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=subtype)
    writer.add_string("hybrid.base_model", args.base.name)
    writer.add_string("hybrid.donor_model", args.donor.name)
    writer.add_string("hybrid.replaced_layers", ",".join(map(str, sorted(layers))))

    selected = []
    base_bytes = donor_bytes = 0
    tensors_to_write = []
    for base_tensor in base.tensors:
        donor_tensor = donor_tensors[base_tensor.name]
        source = donor_tensor if use_donor(base_tensor.name, layers) else base_tensor
        tensors_to_write.append(source)
        if source is donor_tensor:
            selected.append(base_tensor.name)
            base_bytes += base_tensor.n_bytes
            donor_bytes += donor_tensor.n_bytes
        writer.add_tensor_info(source.name, source.data.shape, source.data.dtype,
                               source.data.nbytes, source.tensor_type)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for index, tensor in enumerate(tensors_to_write):
        writer.write_tensor_data(tensor.data, tensor_endianess=(
            donor.endianess if use_donor(tensor.name, layers) else base.endianess))
        if (index + 1) % 50 == 0:
            print(f"wrote {index + 1}/{len(tensors_to_write)} tensors", flush=True)
    writer.close()

    # Reopen and prove that every output tensor has the intended raw payload.
    result = gguf.GGUFReader(args.output, "r")
    result_tensors = {tensor.name: tensor for tensor in result.tensors}
    if result_tensors.keys() != base_tensors.keys():
        raise RuntimeError("output tensor names changed during write")
    for name, result_tensor in result_tensors.items():
        expected = donor_tensors[name] if use_donor(name, layers) else base_tensors[name]
        if result_tensor.tensor_type != expected.tensor_type:
            raise RuntimeError(f"output tensor type mismatch for {name}")
        if digest(result_tensor.data) != digest(expected.data):
            raise RuntimeError(f"output tensor bytes mismatch for {name}")
    added = donor_bytes - base_bytes
    print(f"verified {len(result_tensors)} tensors; replaced {len(selected)} tensors in "
          f"layers {sorted(layers)}; tensor payload change {added / 2**20:+.1f} MiB; "
          f"output size {os.path.getsize(args.output) / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
