#!/usr/bin/env python3
"""Lightweight tests for selective-precision reporting helpers."""

import tempfile
import unittest
from pathlib import Path

from analyze_hybrid_precision import write_report


class HybridPrecisionTest(unittest.TestCase):
    def test_report_identifies_size_efficient_variant(self) -> None:
        base = {"label": "Q2 baseline", "kind": "base", "model_bytes": 1000,
                "added_bytes": 0, "kl": 1.0, "kl_recovery": 0.0, "js_recovery": 0.0,
                "top1_agreement": 0.5, "top10_overlap": 0.5,
                "chunk_interval_low": 0.0, "chunk_interval_high": 0.0,
                "improved_chunks": 0, "marginal_kl_per_mib": ""}
        variant = {**base, "label": "one block", "kind": "hybrid",
                   "model_bytes": 1100, "added_bytes": 100, "kl": 0.9,
                   "kl_recovery": 0.1, "js_recovery": 0.08,
                   "improved_chunks": 24, "chunk_interval_low": 0.01,
                   "chunk_interval_high": 0.02, "marginal_kl_per_mib": 0.001}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, [base, variant])
            text = path.read_text()
        self.assertIn("most size-efficient", text)
        self.assertIn("one block", text)


if __name__ == "__main__":
    unittest.main()
