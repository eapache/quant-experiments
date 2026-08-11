#!/usr/bin/env python3
"""Small deterministic tests for sparse output-head correction."""

import unittest

import numpy as np

from analyze_output_head import (centered_rmse, compute_corrections, evaluate,
                                 top_indices)
from analyze_sparse import PairedRow


class OutputHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            PairedRow(np.array([2, 5, 7], dtype=np.int32),
                      np.array([0.0, 2.0, 1.0], dtype=np.float32),
                      np.array([0.0, 1.0, 2.0], dtype=np.float32),
                      np.ones(3, dtype=bool)),
            PairedRow(np.array([1, 5, 9], dtype=np.int32),
                      np.array([2.0, 1.0, 0.0], dtype=np.float32),
                      np.array([1.0, 2.0, 0.0], dtype=np.float32),
                      np.ones(3, dtype=bool)),
        ]

    def test_top_indices_are_candidate_sorted(self) -> None:
        indices, tokens = top_indices(self.rows, 2)
        self.assertEqual(indices[0].tolist(), [2, 1])
        self.assertEqual(indices[1].tolist(), [1, 0])
        self.assertEqual(tokens.tolist(), [1, 5, 7])

    def test_weight_delta_corrects_selected_logits(self) -> None:
        indices, tokens = top_indices(self.rows, 2)
        hidden = np.array([[1.0], [-1.0]], dtype=np.float32)
        # Unique rows 1, 5, 7 respectively. The shared token's correction changes by context.
        delta = np.array([[-1.0], [1.0], [-1.0]], dtype=np.float32)
        corrections = compute_corrections(self.rows, hidden, indices, tokens, delta)
        selected = np.ones(2, dtype=bool)
        raw = evaluate(self.rows, selected, indices, corrections, 0, 0.0)
        fixed = evaluate(self.rows, selected, indices, corrections, 2, 1.0)
        self.assertLess(fixed["kl"], 1e-8)
        self.assertGreater(raw["kl"], 0.1)

    def test_centered_rmse_ignores_logit_offset(self) -> None:
        left = np.array([1.0, 2.0, 4.0])
        self.assertAlmostEqual(centered_rmse(left, left + 100.0), 0.0)


if __name__ == "__main__":
    unittest.main()
