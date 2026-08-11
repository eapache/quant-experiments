#!/usr/bin/env python3
"""Tests for layer-state parsing and drift metrics."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_layer_drift import (check_alignment, heldout_bias_metrics,
                                 layer_metrics, read_layer_states)


class LayerDriftTest(unittest.TestCase):
    def test_read_layer_states(self) -> None:
        n_ctx, n_embd, n_chunks, first, rows, n_layers = 4, 2, 1, 2, 1, 2
        layers = np.array([0, 3], dtype="<u4")
        tokens = np.arange(4, dtype="<i4")
        values = np.arange(4, dtype="<f4").reshape(1, 2, 1, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.bin"
            with path.open("wb") as stream:
                stream.write(b"_layers_")
                stream.write(struct.pack("<7I", 1, n_ctx, n_embd, n_chunks,
                                         first, rows, n_layers))
                stream.write(layers.tobytes())
                stream.write(tokens.tobytes())
                stream.write(values.tobytes())
            loaded, loaded_layers, loaded_tokens, metadata = read_layer_states(path)
            np.testing.assert_array_equal(loaded, values)
            np.testing.assert_array_equal(loaded_layers, layers)
            np.testing.assert_array_equal(loaded_tokens, tokens)
            self.assertEqual(metadata["n_embd"], 2)

    def test_alignment_rejects_different_layers(self) -> None:
        metadata = {"n_ctx": 4}
        with self.assertRaisesRegex(ValueError, "layer selections"):
            check_alignment(np.array([0]), np.array([1]), np.arange(4), np.arange(4),
                            metadata, metadata)

    def test_metrics_and_heldout_bias(self) -> None:
        reference = np.ones((4, 2, 3), dtype=np.float32)
        candidate = reference - np.array([0.5, 0.0, -0.5], dtype=np.float32)
        metrics = layer_metrics(reference, candidate, np.arange(8, dtype=np.float64))
        self.assertGreater(metrics["relative_error"], 0.0)
        self.assertGreater(metrics["mean_delta_error_fraction"], 0.99)
        folds = heldout_bias_metrics(reference, candidate, 2)
        self.assertEqual(len(folds), 2)
        self.assertTrue(all(row["recovered"] > 0.999 for row in folds))


if __name__ == "__main__":
    unittest.main()
