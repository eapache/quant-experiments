#!/usr/bin/env python3
"""Focused regression tests for monotonic gap calibration."""

import unittest

import numpy as np

from analyze_gap_calibration import (GapModel, MonotonicMap, correct_logits,
                                     fit_monotonic, predict)
from analyze_sparse import PairedRow


class GapCalibrationTest(unittest.TestCase):
    def test_pool_adjacent_violators_handles_fully_decreasing_targets(self) -> None:
        x = np.arange(16, dtype=np.float32)
        model = fit_monotonic(x, -x, np.ones_like(x), knots=8)
        values = predict(model, np.linspace(0, 15, 101))
        self.assertTrue(np.all(np.diff(model.x) > 0) or len(model.x) == 1)
        self.assertTrue(np.all(np.diff(values) >= -1e-7))

    def test_identity_mapping_preserves_logits(self) -> None:
        candidate = np.array([3.0, 1.5, 0.5, -1.0], dtype=np.float32)
        row = PairedRow(
            token_ids=np.arange(4, dtype=np.int32),
            reference=candidate.copy(),
            candidate=candidate,
            exact=np.ones(4, dtype=bool),
        )
        mapping = MonotonicMap(
            np.array([0.0, 8.0], dtype=np.float32),
            np.array([0.0, 8.0], dtype=np.float32),
        )
        model = GapModel(head=4, rank_bins=1, mappings=[mapping])
        bias = np.zeros(4, dtype=np.float32)
        for pairwise in (False, True):
            corrected = correct_logits(row, 1.0, bias, model, 1.0, pairwise)
            np.testing.assert_allclose(corrected, candidate, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
