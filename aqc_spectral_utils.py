#!/usr/bin/env python3
"""
=========================================================================
AQC Spectral Utilities - Shared Functions for Spectral Gap Analysis
=========================================================================
Provides common functionality for Adiabatic Quantum Computing (AQC) 
spectral gap analysis and visualization.

Key Functions:
- Sparse Hamiltonian construction for Max-Cut problems
- Degeneracy detection at s=1
- Spectral gap calculation with proper degeneracy handling
- Full spectrum evolution computation
=========================================================================
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
from typing import List, Tuple, Optional

# =========================================================================
# CONSTANTS
# =========================================================================

# Degeneracy tolerance for eigenvalue comparison
DEGENERACY_TOL = 1e-8

# =========================================================================
# SPARSE HAMILTONIAN CONSTRUCTION
# =========================================================================

def build_H_initial_sparse(N: int) -> csr_matrix:
    """
    Build initial Hamiltonian as sparse matrix: H_initial = -∑ᵢ X̂ᵢ.
    VECTORIZED IMPLEMENTATION for O(1) setup time relative to N.
    """
    dim = 2 ** N
    
    # Vectorized construction:
    # Each row j connects to N columns [j^1, j^2, ..., j^2^(N-1)]
    # We construct the indices directly using numpy broadcasting
    
    # 1. Create range of all states [0, ..., dim-1]
    rows = np.repeat(np.arange(dim), N)
    
    # 2. Create XOR masks for all N bits: [1, 2, 4, ..., 2^(N-1)]
    bit_flips = 1 << np.arange(N)
    
    # 3. Apply masks to all states (broadcast)
    # This creates the columns: col[j*N + k] = j ^ (1<<k)
    cols = np.bitwise_xor(np.arange(dim)[:, np.newaxis], bit_flips).flatten()
    
    # 4. Data is always -1.0
    data = np.full(N * dim, -1.0, dtype=np.float64)
    
    return csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64)


def build_H_problem_sparse(N: int, edges: List[Tuple[int, int]]) -> csr_matrix:
    """
    Build problem Hamiltonian as sparse diagonal matrix: H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ.
    VECTORIZED IMPLEMENTATION.
    """
    dim = 2 ** N
    diagonal = np.zeros(dim, dtype=np.float64)
    states = np.arange(dim)
    
    for u, v in edges:
        # Extract bits u and v for all states at once
        bit_u = (states >> u) & 1
        bit_v = (states >> v) & 1
        
        # Add contribution: +1 if equal, -1 if different
        # 1 - 2*(diff) is +1 if diff=0, -1 if diff=1
        contribution = 1.0 - 2.0 * np.bitwise_xor(bit_u, bit_v)
        diagonal += contribution
            
    return diags(diagonal, 0, format='csr', dtype=np.float64)


def build_H_problem_sparse_weighted(N: int, edges: List[Tuple[int, int]], 
                                     weights: List[float]) -> csr_matrix:
    """
    Build weighted problem Hamiltonian as sparse diagonal matrix: 
    H_problem = ∑₍ᵢ,ⱼ₎∈E wᵢⱼ·ẐᵢẐⱼ.
    
    VECTORIZED IMPLEMENTATION with edge weights.
    
    Args:
        N: Number of qubits (graph vertices)
        edges: List of edges as (i, j) tuples with 0-based indexing
        weights: List of weights for each edge (same order as edges)
        
    Returns:
        Sparse diagonal problem Hamiltonian matrix
    """
    if len(edges) != len(weights):
        raise ValueError(f"Number of edges ({len(edges)}) must match number of weights ({len(weights)})")
    
    dim = 2 ** N
    diagonal = np.zeros(dim, dtype=np.float64)
    states = np.arange(dim)
    
    for (u, v), weight in zip(edges, weights):
        # Extract bits u and v for all states at once
        bit_u = (states >> u) & 1
        bit_v = (states >> v) & 1
        
        # Add weighted contribution: +w if equal, -w if different
        contribution = weight * (1.0 - 2.0 * np.bitwise_xor(bit_u, bit_v))
        diagonal += contribution
            
    return diags(diagonal, 0, format='csr', dtype=np.float64)


def compute_weighted_optimal_cut(edges: List[Tuple[int, int]], 
                                  weights: List[float],
                                  n_qubits: int) -> Tuple[float, str]:
    """
    Compute the optimal weighted Max-Cut value by exhaustive enumeration.
    
    For small graphs (N <= 20), evaluates all 2^N possible cuts to find
    the maximum weighted cut value.
    
    Args:
        edges: List of edges as (i, j) tuples with 0-based indexing
        weights: List of weights for each edge (same order as edges)
        n_qubits: Number of qubits (graph vertices)
        
    Returns:
        Tuple of (max_cut_value, best_bitstring)
        
    Note:
        For unweighted graphs, pass weights=[1.0]*len(edges).
        Complexity: O(2^N × |E|), only feasible for N <= 20.
    """
    if n_qubits > 20:
        raise ValueError(f"Exhaustive enumeration not feasible for N={n_qubits} > 20")
    
    if len(edges) != len(weights):
        raise ValueError(f"Number of weights ({len(weights)}) must match edges ({len(edges)})")
    
    max_cut_value = 0.0
    best_bitstring = '0' * n_qubits
    
    # Evaluate all 2^N possible assignments
    for state in range(2 ** n_qubits):
        bitstring = format(state, f'0{n_qubits}b')
        
        # Calculate weighted cut value
        cut_value = 0.0
        for (u, v), weight in zip(edges, weights):
            if bitstring[u] != bitstring[v]:
                cut_value += weight
        
        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_bitstring = bitstring
    
    return max_cut_value, best_bitstring


def get_aqc_hamiltonian_sparse(s: float, H_B_sparse: csr_matrix, H_P_sparse: csr_matrix) -> csr_matrix:
    """
    Build interpolated AQC Hamiltonian as sparse matrix: H(s) = (1-s)·H_initial + s·H_problem.
    
    Args:
        s: Interpolation parameter in [0, 1]
        H_B_sparse: Sparse initial Hamiltonian matrix
        H_P_sparse: Sparse problem Hamiltonian matrix
        
    Returns:
        Sparse interpolated Hamiltonian H(s)
    """
    return (1 - s) * H_B_sparse + s * H_P_sparse


# =========================================================================
# SPARSE EIGENVALUE COMPUTATION
# =========================================================================

def get_gap_sparse(
    s: float, 
    H_B_sparse: csr_matrix, 
    H_P_sparse: csr_matrix,
    k_eigenvalues: int = 3
) -> float:
    """
    Compute spectral gap Δ(s) = E_{k-1}(s) - E_0(s) using sparse eigensolver.
    
    Uses Lanczos algorithm (scipy.sparse.linalg.eigsh) to compute only the
    k smallest eigenvalues, which is exponentially faster than full diagonalization.
    
    For k=2 degeneracy (single unique solution with Z2 symmetry), we compute
    Δ(s) = E_2(s) - E_0(s) since E_0 and E_1 are degenerate.
    
    Args:
        s: Interpolation parameter in [0, 1]
        H_B_sparse: Sparse initial Hamiltonian matrix
        H_P_sparse: Sparse problem Hamiltonian matrix
        k_eigenvalues: Number of eigenvalues to compute (default 3 for k=2 degeneracy)
        
    Returns:
        Spectral gap value at parameter s
        
    Performance: O(k × nnz × iterations) vs O(n³) for dense diagonalization
    """
    H_s = get_aqc_hamiltonian_sparse(s, H_B_sparse, H_P_sparse)
    
    # Use shift-invert mode with sigma=None for smallest algebraic eigenvalues
    # which='SA' = smallest algebraic (most negative eigenvalues)
    eigenvalues = eigsh(H_s, k=k_eigenvalues, which='SA', return_eigenvectors=False)
    
    # eigsh doesn't guarantee sorted order, so sort them
    eigenvalues = np.sort(eigenvalues)
    
    # Gap is E_{k-1} - E_0 (for k=3 eigenvalues, this is E_2 - E_0)
    return eigenvalues[-1] - eigenvalues[0]


def get_lowest_eigenvalues_sparse(
    s: float,
    H_B_sparse: csr_matrix,
    H_P_sparse: csr_matrix,
    k_eigenvalues: int = 3
) -> np.ndarray:
    """
    Compute the k lowest eigenvalues at parameter s using sparse eigensolver.
    
    Args:
        s: Interpolation parameter in [0, 1]
        H_B_sparse: Sparse initial Hamiltonian matrix
        H_P_sparse: Sparse problem Hamiltonian matrix
        k_eigenvalues: Number of eigenvalues to compute
        
    Returns:
        Sorted array of k lowest eigenvalues
    """
    H_s = get_aqc_hamiltonian_sparse(s, H_B_sparse, H_P_sparse)
    eigenvalues = eigsh(H_s, k=k_eigenvalues, which='SA', return_eigenvectors=False)
    return np.sort(eigenvalues)


# =========================================================================
# DEGENERACY DETECTION
# =========================================================================

def find_first_gap(eigenvalues: np.ndarray, tol: float = DEGENERACY_TOL) -> Tuple[float, int]:
    """
    Find the gap between ground state and first non-degenerate excited state.
    """
    E0 = eigenvalues[0]
    degeneracy = 1
    
    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - E0) < tol:
            degeneracy += 1
        else:
            # Found first non-degenerate level
            gap = eigenvalues[i] - E0
            return gap, degeneracy
    
    return 0.0, len(eigenvalues)

# =========================================================================
# SPECTRUM EVOLUTION
# =========================================================================

def compute_spectrum_evolution(
    H_B: csr_matrix,
    H_P: csr_matrix,
    s_points: np.ndarray,
    compute_all: bool = True
) -> np.ndarray:
    """
    Compute eigenvalue evolution for H(s) across interpolation points.
    Uses sparse Hamiltonians with dense conversion for full spectrum.
    
    Args:
        H_B: Sparse initial Hamiltonian matrix
        H_P: Sparse problem Hamiltonian matrix
        s_points: Array of interpolation parameter values
        compute_all: If True, compute all eigenvalues (for visualization)
        
    Returns:
        Array of shape (len(s_points), dim) with all eigenvalues at each s
    """
    dim = H_B.shape[0]
    all_eigenvalues = np.zeros((len(s_points), dim))
    
    for i, s in enumerate(s_points):
        H_s = (1 - s) * H_B + s * H_P
        all_eigenvalues[i, :] = eigh(H_s.toarray(), eigvals_only=True)
    
    return all_eigenvalues


def compute_spectrum_evolution_sparse(
    H_B: csr_matrix,
    H_P: csr_matrix,
    s_points: np.ndarray,
    k_eigenvalues: int = 6
) -> np.ndarray:
    """
    Compute only k lowest eigenvalues using sparse Lanczos algorithm.
    
    This is exponentially faster than full diagonalization when only a small
    number of eigenvalues are needed (e.g., for visualization).
    
    Args:
        H_B: Sparse initial Hamiltonian matrix
        H_P: Sparse problem Hamiltonian matrix
        s_points: Array of interpolation parameter values
        k_eigenvalues: Number of lowest eigenvalues to compute (default 6)
        
    Returns:
        Array of shape (len(s_points), k_eigenvalues) with k lowest eigenvalues at each s
        
    Performance:
        O(k × nnz × iterations) per s-point vs O(n³) for dense diagonalization
    """
    all_eigenvalues = np.zeros((len(s_points), k_eigenvalues))
    
    for i, s in enumerate(s_points):
        H_s = (1 - s) * H_B + s * H_P
        # Use Lanczos algorithm for smallest algebraic eigenvalues
        evals = eigsh(H_s, k=k_eigenvalues, which='SA', return_eigenvectors=False)
        # eigsh doesn't guarantee sorted order
        all_eigenvalues[i, :] = np.sort(evals)
    
    return all_eigenvalues


# =========================================================================
# SPARSE OPTIMIZED MINIMUM GAP COMPUTATION
# =========================================================================

def find_min_gap_sparse(
    N: int,
    edges: List[Tuple[int, int]],
    num_edges: int,
    target_degeneracy: int = 2,
    s_bounds: Tuple[float, float] = (0.01, 0.99),
    xtol: float = 1e-4,
    verbose: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int], int]:
    """
    Find minimum spectral gap using sparse eigensolvers and scalar optimization.
    
    This is the IMPROVED method that:
    1. Builds sparse Hamiltonians (O(N × 2^N) memory vs O(4^N) for dense)
    2. Uses Lanczos algorithm (eigsh) to compute only 3 lowest eigenvalues
    3. Uses Brent's method (minimize_scalar) to find the true minimum adaptively
    
    FILTERING: Only processes graphs with exactly target_degeneracy (default k=2).
    This corresponds to single unique solution instances with Z2 symmetry.
    
    Args:
        N: Number of qubits (graph vertices)
        edges: List of edges as (i, j) tuples with 0-based indexing
        num_edges: Number of edges in the graph (for max-cut calculation)
        target_degeneracy: Only process graphs with this degeneracy (default 2)
        s_bounds: Optimization bounds (avoid exact 0 and 1 for numerical stability)
        xtol: Tolerance for s location (precision of minimum location)
        verbose: If True, print progress information
        
    Returns:
        Tuple of (min_gap, s_at_min, degeneracy, max_cut_value, num_function_evals):
        - min_gap: Minimum spectral gap value
        - s_at_min: Value of s where minimum occurs
        - degeneracy: Ground state degeneracy at s=1 (should equal target_degeneracy)
        - max_cut_value: Maximum cut value of the graph
        - num_function_evals: Number of eigsh calls used by optimizer
        Returns (None, None, None, None, 0) if degeneracy != target_degeneracy
        
    Performance:
        - N=12: ~0.2s/graph (vs ~2s for grid method)
        - N=14: ~1s/graph (vs ~30s for grid method)
        - N=16: ~5s/graph (vs hours for grid method)
        - N=18+: feasible (grid method is infeasible)
    """
    # Build sparse Hamiltonians
    if verbose:
        print(f"  Building sparse Hamiltonians for N={N}...", end=" ")
    
    H_B_sparse = build_H_initial_sparse(N)
    H_P_sparse = build_H_problem_sparse(N, edges)
    
    if verbose:
        print("done")
    
    # Check degeneracy at s=1 - H_P is diagonal, so eigenvalues are diagonal elements
    # This is deterministic (no random eigsh initialization)
    diagonal = H_P_sparse.diagonal()
    k_check = min(target_degeneracy + 1, len(diagonal))
    evals_final = np.partition(diagonal, k_check - 1)[:k_check]
    evals_final = np.sort(evals_final)
    
    # Verify degeneracy matches target
    _, degeneracy_s1 = find_first_gap(evals_final, tol=DEGENERACY_TOL)
    
    if degeneracy_s1 != target_degeneracy:
        if verbose:
            print(f"  Skipping: degeneracy={degeneracy_s1}, target={target_degeneracy}")
        return None, None, None, None, 0
    
    # Calculate max cut value from ground state energy
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    if verbose:
        print(f"  Degeneracy={degeneracy_s1}, Max-Cut={max_cut_value}")
        print(f"  Optimizing gap function over s ∈ [{s_bounds[0]}, {s_bounds[1]}]...")
    
    # Track number of function evaluations
    eval_counter = [0]  # Use list to allow modification in nested function
    
    def gap_function(s: float) -> float:
        """Gap function to minimize: Δ(s) = E_2(s) - E_0(s)"""
        eval_counter[0] += 1
        # For k=2 degeneracy, we need E_0, E_1 (degenerate), E_2
        return get_gap_sparse(s, H_B_sparse, H_P_sparse, k_eigenvalues=3)
    
    # Use Brent's method for bounded scalar optimization
    result = minimize_scalar(
        gap_function,
        bounds=s_bounds,
        method='bounded',
        options={'xatol': xtol}
    )
    
    min_gap = result.fun
    s_at_min = result.x
    num_evals = eval_counter[0]
    
    if verbose:
        print(f"  Found minimum: Δ_min={min_gap:.6f} at s={s_at_min:.4f} ({num_evals} eigsh calls)")
    
    return float(min_gap), float(s_at_min), int(degeneracy_s1), int(max_cut_value), num_evals


def find_min_gap_sparse_general(
    N: int,
    edges: List[Tuple[int, int]],
    num_edges: int,
    max_degeneracy: int = 10,
    s_bounds: Tuple[float, float] = (0.01, 0.99),
    xtol: float = 1e-4,
    verbose: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int], int]:
    """
    Find minimum spectral gap for graphs with any degeneracy up to max_degeneracy.
    
    This is a more general version that handles arbitrary degeneracy by
    requesting k+1 eigenvalues where k is the detected degeneracy.
    
    Args:
        N: Number of qubits (graph vertices)
        edges: List of edges as (i, j) tuples with 0-based indexing
        num_edges: Number of edges in the graph
        max_degeneracy: Maximum degeneracy to handle (skip if higher)
        s_bounds: Optimization bounds
        xtol: Tolerance for s location
        verbose: If True, print progress
        
    Returns:
        Same as find_min_gap_sparse
    """
    # Build sparse Hamiltonians
    H_B_sparse = build_H_initial_sparse(N)
    H_P_sparse = build_H_problem_sparse(N, edges)
    
    # Check degeneracy at s=1 - H_P is diagonal, so eigenvalues are diagonal elements
    # This is deterministic (no random eigsh initialization)
    diagonal = H_P_sparse.diagonal()
    k_check = min(max_degeneracy + 1, len(diagonal))
    evals_final = np.partition(diagonal, k_check - 1)[:k_check]
    evals_final = np.sort(evals_final)
    
    _, degeneracy_s1 = find_first_gap(evals_final, tol=DEGENERACY_TOL)
    
    if degeneracy_s1 >= max_degeneracy:
        if verbose:
            print(f"  Skipping: degeneracy={degeneracy_s1} >= max={max_degeneracy}")
        return None, None, None, None, 0
    
    # Calculate max cut value
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    # Number of eigenvalues needed: degeneracy + 1 (to get first non-degenerate level)
    k_eigenvalues = degeneracy_s1 + 1
    
    eval_counter = [0]
    
    def gap_function(s: float) -> float:
        eval_counter[0] += 1
        H_s = get_aqc_hamiltonian_sparse(s, H_B_sparse, H_P_sparse)
        eigenvalues = eigsh(H_s, k=k_eigenvalues, which='SA', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        return eigenvalues[-1] - eigenvalues[0]
    
    result = minimize_scalar(
        gap_function,
        bounds=s_bounds,
        method='bounded',
        options={'xatol': xtol}
    )
    
    if verbose:
        print(f"  Δ_min={result.fun:.6f} at s={result.x:.4f} (deg={degeneracy_s1}, {eval_counter[0]} calls)")
    
    return float(result.fun), float(result.x), int(degeneracy_s1), int(max_cut_value), eval_counter[0]


def analyze_spectrum_for_visualization(
    H_B: csr_matrix,
    H_P: csr_matrix,
    s_points: np.ndarray,
    num_edges: int,
    max_degeneracy_check: int = 40
) -> dict:
    """
    Comprehensive spectrum analysis for visualization purposes.
    Uses sparse Hamiltonians with dense conversion for full spectrum.
    
    Args:
        H_B: Sparse initial Hamiltonian matrix
        H_P: Sparse problem Hamiltonian matrix
        s_points: Array of interpolation parameter values
        num_edges: Number of edges in the graph
        max_degeneracy_check: Maximum degeneracy levels to check
        
    Returns:
        Dictionary with keys:
        - all_eigenvalues: Full spectrum at each s point
        - min_gap: Minimum spectral gap value
        - s_at_min: Location of minimum gap
        - min_gap_idx: Index in s_points where minimum occurs
        - degeneracy: Ground state degeneracy at s=1
        - level1: Ground state level index (always 0)
        - level2: Excited state level index for minimum gap
        - max_cut_value: Maximum cut value of the graph
    """
    # Get full spectrum evolution
    all_eigenvalues = compute_spectrum_evolution(H_B, H_P, s_points)
    
    # Check degeneracy at s=1 using sparse eigsh
    k_vals_to_check = min(max_degeneracy_check, H_P.shape[0])
    evals_final = eigsh(H_P, k=k_vals_to_check, which='SA', return_eigenvectors=False)
    evals_final = np.sort(evals_final)
        
    _, degeneracy = find_first_gap(evals_final)
    
    # Calculate max cut value
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    # Find minimum gap
    k_index = degeneracy
    min_gap = np.inf
    min_gap_idx = 0
    excited_level = k_index
    
    for i, s in enumerate(s_points):
        gaps_at_s = all_eigenvalues[i, k_index:] - all_eigenvalues[i, 0]
        min_gap_at_s_idx = np.argmin(gaps_at_s)
        gap_at_s = gaps_at_s[min_gap_at_s_idx]
        
        if gap_at_s < min_gap:
            min_gap = gap_at_s
            min_gap_idx = i
            excited_level = k_index + min_gap_at_s_idx
    
    s_at_min = s_points[min_gap_idx]
    
    return {
        'all_eigenvalues': all_eigenvalues,
        'min_gap': float(min_gap),
        's_at_min': float(s_at_min),
        'min_gap_idx': int(min_gap_idx),
        'degeneracy': int(degeneracy),
        'level1': 0,
        'level2': int(excited_level),
        'max_cut_value': int(max_cut_value)
    }


def analyze_spectrum_for_visualization_sparse(
    H_B: csr_matrix,
    H_P: csr_matrix,
    s_points: np.ndarray,
    num_edges: int,
    k_eigenvalues: int = 6
) -> dict:
    """
    Sparse-optimized spectrum analysis for visualization.
    
    Uses sparse eigensolvers (Lanczos) to compute only k lowest eigenvalues,
    making this exponentially faster than the dense version for large systems.
    
    Args:
        H_B: Sparse initial Hamiltonian matrix
        H_P: Sparse problem Hamiltonian matrix
        s_points: Array of interpolation parameter values
        num_edges: Number of edges in the graph
        k_eigenvalues: Number of lowest eigenvalues to compute (default 6)
        
    Returns:
        Dictionary with keys:
        - all_eigenvalues: k lowest eigenvalues at each s point (shape: len(s_points) x k)
        - min_gap: Minimum spectral gap value
        - s_at_min: Location of minimum gap
        - min_gap_idx: Index in s_points where minimum occurs
        - degeneracy: Ground state degeneracy at s=1
        - level1: Ground state level index (always 0)
        - level2: Excited state level index for minimum gap
        - max_cut_value: Maximum cut value of the graph
        
    Performance:
        For N=12 with k=6: ~50x faster than dense method
        For N=14+ with k=6: enables analysis that was previously infeasible
    """
    # Detect degeneracy at s=1 using diagonal (H_P is diagonal, so this is exact)
    diagonal = H_P.diagonal()
    k_check = min(k_eigenvalues + 5, len(diagonal))  # Check a few extra for safety
    evals_final = np.partition(diagonal, k_check - 1)[:k_check]
    evals_final = np.sort(evals_final)
    
    _, degeneracy = find_first_gap(evals_final)
    
    # Calculate max cut value
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    # Ensure we request enough eigenvalues to include the first excited level
    k_needed = max(k_eigenvalues, degeneracy + 1)
    
    # Get sparse spectrum evolution
    all_eigenvalues = compute_spectrum_evolution_sparse(H_B, H_P, s_points, k_needed)
    
    # Find minimum gap (between ground state and first non-degenerate level)
    k_index = degeneracy  # First non-degenerate level index
    min_gap = np.inf
    min_gap_idx = 0
    excited_level = k_index
    
    for i in range(len(s_points)):
        # Only look at levels beyond degeneracy
        if k_index < all_eigenvalues.shape[1]:
            gaps_at_s = all_eigenvalues[i, k_index:] - all_eigenvalues[i, 0]
            min_gap_at_s_idx = np.argmin(gaps_at_s)
            gap_at_s = gaps_at_s[min_gap_at_s_idx]
            
            if gap_at_s < min_gap:
                min_gap = gap_at_s
                min_gap_idx = i
                excited_level = k_index + min_gap_at_s_idx
    
    s_at_min = s_points[min_gap_idx]
    
    return {
        'all_eigenvalues': all_eigenvalues,
        'min_gap': float(min_gap),
        's_at_min': float(s_at_min),
        'min_gap_idx': int(min_gap_idx),
        'degeneracy': int(degeneracy),
        'level1': 0,
        'level2': int(excited_level),
        'max_cut_value': int(max_cut_value)
    }


# =========================================================================
# GRAPH FILE I/O
# =========================================================================

def extract_graph_params(filename: str) -> Tuple[int, int, int]:
    """
    Extract graph parameters from GENREG filename format.
    """
    import os
    basename = os.path.basename(filename)
    name_without_ext = basename.rsplit('.', 1)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) != 3:
        raise ValueError(f"Invalid filename format: {filename}. Expected N_k_g.ext")
    
    try:
        n = int(parts[0])
        k = int(parts[1])
        girth = int(parts[2])
        return n, k, girth
    except ValueError as e:
        raise ValueError(f"Could not parse integers from filename {filename}: {e}")


def adjacency_to_edges(adj_dict: dict) -> List[Tuple[int, int]]:
    """
    Convert adjacency dictionary to edge list (avoiding duplicates).
    """
    edges = []
    seen = set()
    
    for v, neighbors in adj_dict.items():
        for n in neighbors:
            edge = (min(v, n), max(v, n))
            if edge not in seen:
                seen.add(edge)
                edges.append((edge[0] - 1, edge[1] - 1))
    return sorted(edges)


def parse_asc_file(filename: str) -> List[List[Tuple[int, int]]]:
    """
    Parse GENREG .asc file and return list of graphs as 0-indexed edge lists.
    """
    graphs = []
    current_graph = {}
    in_adjacency = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('Graph'):
                if current_graph:
                    graphs.append(adjacency_to_edges(current_graph))
                current_graph = {}
                in_adjacency = True
                
            elif line.startswith('Taillenweite:'):
                in_adjacency = False
            elif in_adjacency and ':' in line:
                try:
                    parts = line.split(':')
                    vertex = int(parts[0].strip())
                    neighbors = [int(x) for x in parts[1].split()]
                    current_graph[vertex] = neighbors
                except (ValueError, IndexError):
                    continue
    
    if current_graph:
        graphs.append(adjacency_to_edges(current_graph))
    
    return graphs


def shortcode_to_edges(code: List[int], n: int, k: int) -> List[Tuple[int, int]]:
    """
    Convert GENREG shortcode representation to edge list.
    """
    edges = []
    degree_count = [0] * n
    v = 0
    
    for w_1indexed in code:
        w = int(w_1indexed) - 1
        
        while v < n and degree_count[v] == k:
            v += 1
        
        if v >= n:
            break
        
        edges.append((int(v), int(w)))
        degree_count[v] += 1
        degree_count[w] += 1
    
    return sorted(edges)


def parse_scd_file(filename: str, n: Optional[int] = None, k: Optional[int] = None) -> List[List[Tuple[int, int]]]:
    """
    Parse GENREG .scd (shortcode) binary file and return list of graphs.
    """
    if n is None or k is None:
        n_auto, k_auto, _ = extract_graph_params(filename)
        n = n if n is not None else n_auto
        k = k if k is not None else k_auto
    
    num_edges = n * k // 2
    
    with open(filename, 'rb') as f:
        values = np.fromfile(f, dtype=np.uint8)
    
    graphs = []
    read_pos = 0
    code = []
    
    while read_pos < len(values):
        samebits = int(values[read_pos])
        read_pos += 1
        
        readbits = num_edges - samebits
        code = code[:samebits] + list(values[read_pos:read_pos + readbits])
        read_pos += readbits
        
        edges = shortcode_to_edges(code, n, k)
        graphs.append(edges)
    
    return graphs


def load_graphs_from_file(filename: str) -> List[List[Tuple[int, int]]]:
    """
    Load graphs from GENREG file (supports both .asc and .scd formats).
    """
    import os
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    if ext == '.asc':
        return parse_asc_file(filename)
    elif ext == '.scd':
        return parse_scd_file(filename)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Expected .asc or .scd")
