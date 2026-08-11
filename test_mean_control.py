#!/usr/bin/env python3
"""Tests for residual-stream mean control vectors."""

import unittest

import numpy as np

from build_mean_control_vectors import mean_vector


class MeanControlTest(unittest.TestCase):
    def test_mean_vector_uses_only_selected_chunks(self) -> None:
        candidate = np.zeros((3, 2, 2), dtype=np.float32)
        reference = candidate.copy()
        reference[0] = [1.0, 3.0]
        reference[1] = [5.0, 7.0]
        reference[2] = [100.0, 200.0]
        vector = mean_vector(reference, candidate, np.array([0, 1]))
        np.testing.assert_allclose(vector, [3.0, 5.0])


if __name__ == "__main__":
    unittest.main()
