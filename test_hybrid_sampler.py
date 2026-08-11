#!/usr/bin/env python3
"""Tests for compact-hybrid temperature reporting."""

import tempfile
import unittest
from pathlib import Path

from analyze_hybrid_sampler import write_report


class HybridSamplerTest(unittest.TestCase):
    def test_report_rejects_unproven_specific_temperature(self) -> None:
        rows = []
        for fold in range(4):
            for method, js, temperature in (
                ("same-settings", 0.05, 0.8),
                ("q2-frozen-temperature", 0.049, 0.7625),
                ("hybrid-tuned-temperature", 0.0489, 0.77),
            ):
                rows.append({"fold": fold, "method": method, "sampler_js": js,
                             "sampler_tv": 0.15,
                             "candidate_temperature": temperature})
        summary = {
            "passes_hybrid_specific_gate": False,
            "fold_temperatures": "0.7700+0.7700+0.7700+0.7700",
            "frozen_hybrid_temperature": 0.77,
            "incrementally_improved_chunks": 18,
            "incremental_mean_js_reduction": 0.0001,
            "incremental_interval_low": -0.0001,
            "incremental_interval_high": 0.0003,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, rows, summary, 0.8, 0.95, 0.7625)
            text = path.read_text()
        self.assertIn("Q2-frozen temperature", text)
        self.assertIn("**fails** the promotion gate", text)


if __name__ == "__main__":
    unittest.main()
