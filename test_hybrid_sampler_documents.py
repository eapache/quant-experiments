#!/usr/bin/env python3
"""Tests for frozen hybrid sampler document reporting."""

import tempfile
import unittest
from pathlib import Path

from evaluate_hybrid_sampler_documents import write_report


class FrozenHybridSamplerDocumentsTest(unittest.TestCase):
    def test_report_includes_document_level_summary(self) -> None:
        row = {
            "document": "code-a", "baseline_js": 0.05, "corrected_js": 0.049,
            "js_recovery": 0.02, "improved_chunks": 20, "chunks": 32,
            "chunk_interval_low": -0.001, "chunk_interval_high": 0.003,
        }
        aggregate = {
            "improved_documents": 1, "documents": 1, "positions": 2016,
            "pooled_baseline_js": 0.05, "pooled_corrected_js": 0.049,
            "pooled_js_recovery": 0.02, "mean_document_js_reduction": 0.001,
            "document_interval_low": -0.001, "document_interval_high": 0.003,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, [row], aggregate, 0.7625, 0.95)
            text = path.read_text()
        self.assertIn("T=0.7625", text)
        self.assertIn("| code-a |", text)
        self.assertIn("Document variation is the primary", text)


if __name__ == "__main__":
    unittest.main()
