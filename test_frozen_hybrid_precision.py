#!/usr/bin/env python3
"""Tests for frozen selective-precision reporting."""

import tempfile
import unittest
from pathlib import Path

from evaluate_frozen_hybrid_precision import write_report


class FrozenHybridPrecisionTest(unittest.TestCase):
    def test_report_preserves_preselected_primary(self) -> None:
        base = {
            "label": "Q2 baseline", "role": "baseline", "is_primary": False, "kl": 1.0,
            "kl_recovery": 0.0, "js_recovery": 0.0, "top1_agreement": 0.5,
            "top10_overlap": 0.5, "improved_chunks": 0,
            "chunk_interval_low": 0.0, "chunk_interval_high": 0.0,
        }
        primary = {
            **base, "label": "early-Q8-2", "role": "preselected",
            "is_primary": True, "kl": 0.9,
            "kl_recovery": 0.1, "js_recovery": 0.08, "improved_chunks": 28,
            "chunk_interval_low": 0.01, "chunk_interval_high": 0.03,
        }
        diagnostic = {
            **primary, "label": "early-Q8-3", "role": "hybrid",
            "is_primary": False, "kl": 0.8,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, [base, primary, diagnostic], "early-Q8-2")
            text = path.read_text()
        self.assertIn("**early-Q8-2** was selected on prose", text)
        self.assertIn("early-Q8-2 | preselected", text)
        self.assertIn("early-Q8-3 | hybrid", text)


if __name__ == "__main__":
    unittest.main()
