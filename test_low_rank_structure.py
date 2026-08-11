import numpy as np

from analyze_low_rank import BasisModel
from analyze_low_rank_structure import gaussian_basis, permuted_basis, randomized_svd_basis


def test_gaussian_basis_is_orthonormal() -> None:
    model = BasisModel(np.arange(40, dtype=np.int32), np.empty((0, 40), dtype=np.float32),
                       np.empty(0, dtype=np.float32))
    basis = gaussian_basis(model, 8, seed=3)
    np.testing.assert_allclose(basis.basis @ basis.basis.T, np.eye(8), atol=2e-6)


def test_permutation_preserves_gram_matrix_and_spectrum() -> None:
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(6, 30)).astype(np.float32)
    q, _ = np.linalg.qr(raw.T, mode="reduced")
    model = BasisModel(np.arange(30, dtype=np.int32), q.T.astype(np.float32),
                       np.arange(6, 0, -1, dtype=np.float32))
    permuted = permuted_basis(model, 6, seed=9)
    np.testing.assert_allclose(permuted.basis @ permuted.basis.T,
                               model.basis @ model.basis.T, atol=2e-6)
    np.testing.assert_array_equal(permuted.singular_values, model.singular_values)


def test_randomized_svd_recovers_exact_low_rank_row_space() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(50, 3)).astype(np.float32)
    right = rng.normal(size=(3, 25)).astype(np.float32)
    matrix = left @ right
    model = randomized_svd_basis(matrix, np.arange(25, dtype=np.int32), 3, seed=11)
    projection = model.basis.T @ model.basis
    relative_error = np.linalg.norm(matrix - matrix @ projection) / np.linalg.norm(matrix)
    assert relative_error < 2e-6
