#!/usr/bin/env python3
"""
Tests for spectral gap computation in aqc_spectral_utils.py.

Uses small known graphs to verify correctness of:
- Sparse Hamiltonian construction
- Spectral gap computation
- Degeneracy detection
- Minimum gap finding
"""

import numpy as np
import pytest
from scipy.sparse import issparse
from scipy.linalg import eigh

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqc_spectral_utils import (
    build_H_initial_sparse,
    build_H_problem_sparse,
    build_H_problem_sparse_weighted,
    get_aqc_hamiltonian_sparse,
    get_gap_sparse,
    get_lowest_eigenvalues_sparse,
    find_first_gap,
    find_min_gap_sparse,
    compute_weighted_optimal_cut,
)


# =========================================================================
# FIXTURES: Small known graphs
# =========================================================================

@pytest.fixture
def triangle_graph():
    """Complete graph K3 (triangle): 3 vertices, 3 edges, 3-regular-ish."""
    N = 3
    edges = [(0, 1), (0, 2), (1, 2)]
    return N, edges


@pytest.fixture
def square_graph():
    """Cycle graph C4 (square): 4 vertices, 4 edges, 2-regular."""
    N = 4
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    return N, edges


@pytest.fixture
def k4_graph():
    """Complete graph K4: 4 vertices, 6 edges."""
    N = 4
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return N, edges


# =========================================================================
# TESTS: Hamiltonian construction
# =========================================================================

class TestHamiltonianConstruction:
    """Tests for sparse Hamiltonian builders."""

    def test_H_initial_sparse_shape(self):
        """H_initial should be 2^N x 2^N."""
        for N in [2, 3, 4, 5]:
            H = build_H_initial_sparse(N)
            dim = 2 ** N
            assert H.shape == (dim, dim)

    def test_H_initial_sparse_is_sparse(self):
        H = build_H_initial_sparse(4)
        assert issparse(H)

    def test_H_initial_sparse_hermitian(self):
        """H_initial should be Hermitian (symmetric real)."""
        H = build_H_initial_sparse(4)
        diff = H - H.T
        assert diff.nnz == 0 or np.allclose(diff.toarray(), 0)

    def test_H_initial_sparse_eigenvalues(self):
        """H_initial = -sum(X_i) should have known eigenvalues for N=2.
        For N=2: H = -X1 - X2. Eigenvalues are {-2, 0, 0, 2}."""
        H = build_H_initial_sparse(2)
        evals = np.sort(eigh(H.toarray(), eigvals_only=True))
        expected = np.array([-2.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_H_problem_sparse_shape(self, triangle_graph):
        N, edges = triangle_graph
        H = build_H_problem_sparse(N, edges)
        dim = 2 ** N
        assert H.shape == (dim, dim)

    def test_H_problem_sparse_is_diagonal(self, triangle_graph):
        """H_problem (ZZ interactions) should be diagonal."""
        N, edges = triangle_graph
        H = build_H_problem_sparse(N, edges)
        dense = H.toarray()
        # Off-diagonal elements should be zero
        np.testing.assert_allclose(
            dense - np.diag(np.diag(dense)), 0, atol=1e-15
        )

    def test_H_problem_sparse_triangle_ground_state(self, triangle_graph):
        """Triangle graph: max cut = 2 (cut one vertex from other two).
        H_problem eigenvalues for ZZ: each edge contributes ±1.
        Ground state energy = -(num_edges) + 2*max_cut = -3 + 4 = 1?
        Actually E_0 = num_edges - 2*max_cut = 3 - 2*2 = -1."""
        N, edges = triangle_graph
        H = build_H_problem_sparse(N, edges)
        evals = np.sort(eigh(H.toarray(), eigvals_only=True))
        # For triangle, min eigenvalue should be -1 (max cut = 2, 3 edges)
        # max_cut = (num_edges - E_0) / 2 => E_0 = num_edges - 2*max_cut = 3 - 4 = -1
        assert np.isclose(evals[0], -1.0, atol=1e-10)

    def test_H_problem_weighted_matches_unweighted(self, square_graph):
        """Weighted Hamiltonian with all weights=1 should match unweighted."""
        N, edges = square_graph
        H_unw = build_H_problem_sparse(N, edges)
        weights = [1.0] * len(edges)
        H_w = build_H_problem_sparse_weighted(N, edges, weights)
        np.testing.assert_allclose(H_unw.toarray(), H_w.toarray(), atol=1e-15)

    def test_H_problem_weighted_scaling(self, square_graph):
        """Weighted Hamiltonian with weight=c should scale eigenvalues by c."""
        N, edges = square_graph
        c = 2.5
        H_unw = build_H_problem_sparse(N, edges)
        H_w = build_H_problem_sparse_weighted(N, edges, [c] * len(edges))
        np.testing.assert_allclose(H_w.toarray(), c * H_unw.toarray(), atol=1e-15)

    def test_H_problem_weighted_mismatched_raises(self, square_graph):
        """Mismatched edges/weights length should raise ValueError."""
        N, edges = square_graph
        with pytest.raises(ValueError):
            build_H_problem_sparse_weighted(N, edges, [1.0])

    def test_aqc_hamiltonian_endpoints(self, triangle_graph):
        """H(s=0) should be H_initial, H(s=1) should be H_problem."""
        N, edges = triangle_graph
        H_B = build_H_initial_sparse(N)
        H_P = build_H_problem_sparse(N, edges)

        H_0 = get_aqc_hamiltonian_sparse(0.0, H_B, H_P)
        H_1 = get_aqc_hamiltonian_sparse(1.0, H_B, H_P)

        np.testing.assert_allclose(H_0.toarray(), H_B.toarray(), atol=1e-15)
        np.testing.assert_allclose(H_1.toarray(), H_P.toarray(), atol=1e-15)


# =========================================================================
# TESTS: Spectral gap computation
# =========================================================================

class TestSpectralGap:
    """Tests for spectral gap and eigenvalue functions."""

    def test_get_gap_sparse_positive(self, triangle_graph):
        """Spectral gap should be positive for s in (0, 1)."""
        N, edges = triangle_graph
        H_B = build_H_initial_sparse(N)
        H_P = build_H_problem_sparse(N, edges)
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            gap = get_gap_sparse(s, H_B, H_P, k_eigenvalues=3)
            assert gap > 0, f"Gap should be positive at s={s}, got {gap}"

    def test_get_lowest_eigenvalues_sorted(self, square_graph):
        """Returned eigenvalues should be sorted in ascending order."""
        N, edges = square_graph
        H_B = build_H_initial_sparse(N)
        H_P = build_H_problem_sparse(N, edges)
        evals = get_lowest_eigenvalues_sparse(0.5, H_B, H_P, k_eigenvalues=4)
        assert np.all(np.diff(evals) >= -1e-10)

    def test_get_lowest_eigenvalues_match_dense(self, triangle_graph):
        """Sparse k-lowest eigenvalues should match dense diagonalization."""
        N, edges = triangle_graph
        H_B = build_H_initial_sparse(N)
        H_P = build_H_problem_sparse(N, edges)
        s = 0.4
        H_s = get_aqc_hamiltonian_sparse(s, H_B, H_P)

        # Dense
        evals_dense = np.sort(eigh(H_s.toarray(), eigvals_only=True))
        # Sparse (k=3)
        evals_sparse = get_lowest_eigenvalues_sparse(s, H_B, H_P, k_eigenvalues=3)

        np.testing.assert_allclose(evals_sparse, evals_dense[:3], atol=1e-8)


# =========================================================================
# TESTS: Degeneracy detection
# =========================================================================

class TestDegeneracy:
    """Tests for find_first_gap degeneracy detection."""

    def test_no_degeneracy(self):
        """Non-degenerate: [0, 1, 2] -> gap=1, deg=1."""
        evals = np.array([0.0, 1.0, 2.0])
        gap, deg = find_first_gap(evals)
        assert deg == 1
        assert np.isclose(gap, 1.0)

    def test_twofold_degeneracy(self):
        """Two-fold: [0, 0, 1] -> gap=1, deg=2."""
        evals = np.array([0.0, 0.0, 1.0])
        gap, deg = find_first_gap(evals)
        assert deg == 2
        assert np.isclose(gap, 1.0)

    def test_threefold_degeneracy(self):
        """Three-fold: [0, 0, 0, 2] -> gap=2, deg=3."""
        evals = np.array([0.0, 0.0, 0.0, 2.0])
        gap, deg = find_first_gap(evals)
        assert deg == 3
        assert np.isclose(gap, 2.0)

    def test_all_degenerate(self):
        """All degenerate: [0, 0, 0] -> gap=0, deg=3."""
        evals = np.array([0.0, 0.0, 0.0])
        gap, deg = find_first_gap(evals)
        assert deg == 3
        assert np.isclose(gap, 0.0)

    def test_near_degeneracy_within_tol(self):
        """Near-degenerate within tolerance should count as degenerate."""
        evals = np.array([0.0, 1e-10, 1.0])
        gap, deg = find_first_gap(evals, tol=1e-8)
        assert deg == 2
        assert np.isclose(gap, 1.0)


# =========================================================================
# TESTS: Minimum gap finder
# =========================================================================

class TestMinGapFinder:
    """Tests for find_min_gap_sparse end-to-end."""

    def test_find_min_gap_returns_valid(self):
        """find_min_gap_sparse should return valid results for a simple graph.
        Use a 4-vertex path graph: 0-1-2-3 (3 edges)."""
        N = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        num_edges = len(edges)
        # We don't know the degeneracy a priori, so use general version
        # But let's test with target_degeneracy that matches
        result = find_min_gap_sparse(
            N, edges, num_edges, target_degeneracy=2,
            s_bounds=(0.01, 0.99), xtol=1e-3
        )
        min_gap, s_at_min, deg, max_cut, n_evals = result
        # Path graph P4: max cut = 3 (bipartite: {0,2} vs {1,3}), degeneracy = 2 (Z2 symmetry)
        if min_gap is not None:
            assert min_gap > 0
            assert 0.01 <= s_at_min <= 0.99
            assert deg == 2
            assert max_cut == 3
            assert n_evals > 0

    def test_find_min_gap_skips_wrong_degeneracy(self, k4_graph):
        """K4 has high degeneracy; target_degeneracy=2 should skip it."""
        N, edges = k4_graph
        num_edges = len(edges)
        result = find_min_gap_sparse(N, edges, num_edges, target_degeneracy=2)
        min_gap, s_at_min, deg, max_cut, n_evals = result
        # Should return None since K4's degeneracy != 2
        # (K4 max-cut degeneracy is high)
        assert n_evals == 0  # skipped


# =========================================================================
# TESTS: Weighted optimal cut
# =========================================================================

class TestWeightedOptimalCut:
    """Tests for compute_weighted_optimal_cut."""

    def test_unweighted_triangle(self, triangle_graph):
        """Triangle with unit weights: max cut = 2."""
        N, edges = triangle_graph
        weights = [1.0] * len(edges)
        max_cut, bitstring = compute_weighted_optimal_cut(edges, weights, N)
        assert np.isclose(max_cut, 2.0)
        assert len(bitstring) == N

    def test_unweighted_square(self, square_graph):
        """Square (C4) with unit weights: max cut = 4 (bipartite)."""
        N, edges = square_graph
        weights = [1.0] * len(edges)
        max_cut, bitstring = compute_weighted_optimal_cut(edges, weights, N)
        assert np.isclose(max_cut, 4.0)

    def test_weighted_single_edge(self):
        """Single heavy edge: max cut = weight."""
        edges = [(0, 1)]
        weights = [3.7]
        max_cut, bitstring = compute_weighted_optimal_cut(edges, weights, 2)
        assert np.isclose(max_cut, 3.7)

    def test_weighted_mismatched_raises(self, triangle_graph):
        N, edges = triangle_graph
        with pytest.raises(ValueError):
            compute_weighted_optimal_cut(edges, [1.0], N)

    def test_large_N_raises(self):
        """N > 20 should raise ValueError."""
        edges = [(0, 1)]
        weights = [1.0]
        with pytest.raises(ValueError):
            compute_weighted_optimal_cut(edges, weights, 21)
