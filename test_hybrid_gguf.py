#!/usr/bin/env python3
"""Tests for selective hybrid tensor routing."""

import unittest

from build_hybrid_gguf import tensor_layer, use_donor


class HybridGgufTest(unittest.TestCase):
    def test_tensor_layer(self) -> None:
        self.assertEqual(tensor_layer("blk.12.ffn_up.weight"), 12)
        self.assertIsNone(tensor_layer("token_embd.weight"))
        self.assertIsNone(tensor_layer("output_norm.weight"))

    def test_donor_selection_is_exact(self) -> None:
        layers = {0, 1, 2}
        self.assertTrue(use_donor("blk.0.attn_qkv.weight", layers))
        self.assertTrue(use_donor("blk.2.ffn_down.weight", layers))
        self.assertFalse(use_donor("blk.3.attn_q.weight", layers))
        self.assertFalse(use_donor("token_embd.weight", layers))


if __name__ == "__main__":
    unittest.main()
