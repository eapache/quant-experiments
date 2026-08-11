#!/usr/bin/env python3
"""Regression tests for sparse distribution metrics."""

import unittest

import numpy as np

from analyze_sparse import PairedRow, metrics


class SparseMetricsTest(unittest.TestCase):
    def test_top10_handles_fewer_than_ten_observable_tokens(self) -> None:
        row = PairedRow(
            token_ids=np.array([2, 5, 7], dtype=np.int32),
            reference=np.array([-0.1, -1.0, -2.0], dtype=np.float32),
            candidate=np.array([-0.2, -0.9, -1.8], dtype=np.float32),
            exact=np.ones(3, dtype=bool),
        )
        result = metrics([row], np.ones(1, dtype=bool), temperature=1.0)
        self.assertEqual(result["top10_overlap"], 1.0)


if __name__ == "__main__":
    unittest.main()
