#!/usr/bin/env python3
"""Tests for paired nested-hybrid reporting."""

import tempfile
import unittest
from pathlib import Path

from evaluate_incremental_precision import write_report


class IncrementalPrecisionTest(unittest.TestCase):
    def test_report_states_failed_gate(self) -> None:
        rows = [{
            "label": label, "role": role, "kl": kl,
            "kl_recovery_from_q2": recovery, "js_recovery_from_q2": recovery,
            "top1_agreement": 0.8, "top10_overlap": 0.75,
        } for label, role, kl, recovery in (
            ("Q2 baseline", "base", 0.2, 0.0),
            ("current", "current", 0.19, 0.05),
            ("extension", "extension", 0.189, 0.055),
        )]
        comparison = {
            "chunks": 32, "incrementally_improved_chunks": 18,
            "incremental_mean_kl_reduction": 0.001,
            "incremental_interval_low": -0.001,
            "incremental_interval_high": 0.003,
            "passes_incremental_gate": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, rows, comparison, "current", "extension", 10.5)
            text = path.read_text()
        self.assertIn("10.5 MiB extension", text)
        self.assertIn("**fails** the predeclared", text)


if __name__ == "__main__":
    unittest.main()
