import struct
from pathlib import Path

import numpy as np

from analyze_hidden_adapter import load_hidden, predict_kernel, prepare_kernel


def test_kernel_ridge_fits_linear_targets() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(40, 12)).astype(np.float32)
    weights = rng.normal(size=(12, 3))
    targets = features @ weights + np.array([0.3, -0.2, 0.7])
    model = prepare_kernel(features, targets)
    predicted = predict_kernel(model, 1e-6, features)
    assert np.max(np.abs(predicted - targets)) < 1e-5


def test_hidden_loader_checks_alignment(tmp_path: Path) -> None:
    capture = tmp_path / "capture.kld"
    hidden = tmp_path / "hidden.bin"
    tokens = np.arange(16, dtype="<i4")
    capture.write_bytes(b"_logits_" + struct.pack("<Iii", 8, 99, 2) + tokens.tobytes())
    states = np.arange(2 * 3 * 5, dtype="<f4").reshape(6, 5)
    hidden.write_bytes(
        b"_hidden_" + struct.pack("<6I", 1, 8, 5, 2, 4, 3)
        + tokens.tobytes() + states.tobytes())
    loaded = load_hidden(hidden, capture)
    np.testing.assert_array_equal(loaded, states)
