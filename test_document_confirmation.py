#!/usr/bin/env python3
"""Tests for multi-document confirmation reporting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_document_confirmation import mean_t_interval, write_report


class DocumentConfirmationTest(unittest.TestCase):
    def test_four_document_t_interval(self) -> None:
        mean, low, high = mean_t_interval(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(mean, 2.5)
        self.assertAlmostEqual(low, 0.4460, places=4)
        self.assertAlmostEqual(high, 4.5540, places=4)

    def test_report_keeps_document_rows(self) -> None:
        row = {
            "document": "doc-a", "positions": 100, "chunks": 32,
            "baseline_kl": 0.2, "candidate_kl": 0.19, "kl_recovery": 0.05,
            "js_recovery": 0.04, "improved_chunks": 24,
            "chunk_interval_low": 0.001, "chunk_interval_high": 0.02,
        }
        aggregate = {
            "improved_documents": 1, "documents": 1, "positions": 100,
            "pooled_baseline_kl": 0.2, "pooled_candidate_kl": 0.19,
            "pooled_kl_recovery": 0.05, "pooled_js_recovery": 0.04,
            "mean_document_kl_reduction": 0.01,
            "document_interval_low": -0.01, "document_interval_high": 0.03,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, [row], aggregate, "fixed-candidate")
            text = path.read_text()
        self.assertIn("fixed-candidate", text)
        self.assertIn("| doc-a |", text)
        self.assertIn("primary uncertainty summary", text)


if __name__ == "__main__":
    unittest.main()
