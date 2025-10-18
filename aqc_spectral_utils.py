#!/usr/bin/env python3
"""
=========================================================================
AQC Spectral Utilities - Shared Functions for Spectral Gap Analysis
=========================================================================
Provides common functionality for Adiabatic Quantum Computing (AQC) 
spectral gap analysis and visualization.

Key Functions:
- Hamiltonian construction for Max-Cut problems
- Degeneracy detection at s=1
- Spectral gap calculation with proper degeneracy handling
- Full spectrum evolution computation
=========================================================================
"""

import numpy as np
from scipy.linalg import eigh
from typing import List, Tuple, Optional

# =========================================================================
# CONSTANTS
# =========================================================================

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

# Degeneracy tolerance for eigenvalue comparison
DEGENERACY_TOL = 1e-8

# =========================================================================
# HAMILTONIAN CONSTRUCTION
# =========================================================================

def pauli_tensor_product(op_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute tensor product of a list of 2x2 matrices.
    
    Args:
        op_list: List of 2x2 numpy arrays (typically Pauli matrices or identity)
        
    Returns:
        Tensor product as 2^n × 2^n matrix where n = len(op_list)
    """
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result


def get_pauli_term(N: int, pauli_type: str, index1: int, index2: int = -1) -> np.ndarray:
    """
    Generate N-qubit Pauli operator: X_i or Z_i⊗Z_j.
    
    Args:
        N: Number of qubits
        pauli_type: 'X' for single-qubit X operator, 'ZZ' for two-qubit ZZ
        index1: First qubit index (0-based)
        index2: Second qubit index for ZZ operator (0-based)
        
    Returns:
        2^N × 2^N operator matrix
    """
    operators = [IDENTITY] * N
    if pauli_type == 'X':
        operators[index1] = SIGMA_X
    elif pauli_type == 'ZZ':
        operators[index1] = SIGMA_Z
        operators[index2] = SIGMA_Z
    return pauli_tensor_product(operators)


def build_H_initial(N: int) -> np.ndarray:
    """
    Build initial Hamiltonian: H_initial = -∑ᵢ X̂ᵢ (transverse field mixer).
    
    Args:
        N: Number of qubits
        
    Returns:
        2^N × 2^N Hamiltonian matrix
    """
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        H_B += get_pauli_term(N, 'X', i)
    return -H_B


def build_H_problem(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build problem Hamiltonian: H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ (Max-Cut Hamiltonian).
    
    Args:
        N: Number of qubits
        edges: List of edges as (i, j) tuples with 0-based indexing
        
    Returns:
        2^N × 2^N Hamiltonian matrix
    """
    H_P = np.zeros((2**N, 2**N), dtype=complex)
    for u, v in edges:
        H_P += get_pauli_term(N, 'ZZ', u, v)
    return H_P


def get_aqc_hamiltonian(s: float, H_B: np.ndarray, H_P: np.ndarray) -> np.ndarray:
    """
    Build interpolated AQC Hamiltonian: H(s) = (1-s)·H_initial + s·H_problem.
    
    Args:
        s: Interpolation parameter in [0, 1]
        H_B: Initial Hamiltonian matrix
        H_P: Problem Hamiltonian matrix
        
    Returns:
        Interpolated Hamiltonian H(s)
    """
    return (1 - s) * H_B + s * H_P

# =========================================================================
# DEGENERACY DETECTION
# =========================================================================

def find_first_gap(eigenvalues: np.ndarray, tol: float = DEGENERACY_TOL) -> Tuple[float, int]:
    """
    Find the gap between ground state and first non-degenerate excited state.
    
    This determines which excited states are degenerate with the ground state.
    In Max-Cut problems, ground state degeneracy at s=1 corresponds to the
    number of optimal solutions.
    
    Args:
        eigenvalues: Sorted array of eigenvalues (ascending order)
        tol: Tolerance for considering eigenvalues degenerate
        
    Returns:
        Tuple of (gap, degeneracy) where:
        - gap: Energy difference to first non-degenerate level
        - degeneracy: Number of eigenvalues degenerate with ground state
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
    
    # All eigenvalues are degenerate (shouldn't happen in practice)
    return 0.0, len(eigenvalues)

# =========================================================================
# SPECTRUM EVOLUTION
# =========================================================================

def compute_spectrum_evolution(
    H_B: np.ndarray,
    H_P: np.ndarray,
    s_points: np.ndarray,
    compute_all: bool = True
) -> np.ndarray:
    """
    Compute eigenvalue evolution for H(s) across interpolation points.
    
    Args:
        H_B: Initial Hamiltonian matrix
        H_P: Problem Hamiltonian matrix
        s_points: Array of s values in [0, 1] to evaluate
        compute_all: If True, compute all eigenvalues; if False, compute a subset
        
    Returns:
        Array of shape (len(s_points), num_eigenvalues) containing eigenvalues
    """
    dim = H_B.shape[0]
    all_eigenvalues = np.zeros((len(s_points), dim))
    
    for i, s in enumerate(s_points):
        H_s = get_aqc_hamiltonian(s, H_B, H_P)
        all_eigenvalues[i, :] = eigh(H_s, eigvals_only=True)
    
    return all_eigenvalues


def find_min_gap_with_degeneracy(
    H_B: np.ndarray,
    H_P: np.ndarray,
    s_points: np.ndarray,
    num_edges: int,
    k_vals_check: int = 5,
    verbose: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    Calculate minimum spectral gap with degeneracy-aware tracking.
    
    METHODOLOGY (following spectral_gap_analysis.py):
    1. Determines ground state degeneracy k at s=1 (problem Hamiltonian)
    2. For each s, finds minimum gap among all non-degenerate excited states:
       min(E[k:] - E[0]) where E[0] is ground state and E[k:] are non-degenerate
    3. Returns the global minimum gap across all s values
    
    This ensures we don't track "fake" gaps from states that are degenerate
    with the ground state at s=1 (i.e., equivalent optimal solutions).
    
    Args:
        H_B: Initial Hamiltonian matrix
        H_P: Problem Hamiltonian matrix
        s_points: Array of s values to sample
        num_edges: Number of edges in the graph (for max-cut calculation)
        k_vals_check: Skip graphs with degeneracy >= this value
        verbose: If True, print progress
        
    Returns:
        Tuple of (min_gap, s_at_min, degeneracy, max_cut_value, excited_level_index):
        - min_gap: Minimum spectral gap value
        - s_at_min: Value of s where minimum occurs
        - degeneracy: Ground state degeneracy at s=1
        - max_cut_value: Maximum cut value of the graph
        - excited_level_index: Which eigenvalue level creates the min gap
        Returns (None, None, None, None, None) if degeneracy exceeds threshold
    """
    # Determine degeneracy at s=1 (problem Hamiltonian)
    k_vals = min(k_vals_check, H_P.shape[0])
    evals_final = eigh(H_P, eigvals_only=True, subset_by_index=(0, k_vals - 1))
    _, degeneracy_s1 = find_first_gap(evals_final)
    
    # Filter: skip graphs where degeneracy exceeds threshold
    if degeneracy_s1 >= k_vals_check:
        return None, None, None, None, None
    
    # Calculate max cut value from ground state energy
    # For H_problem = ∑_{(i,j)∈E} Z_i Z_j:
    # - Edges in cut contribute -1, edges not in cut contribute +1
    # - Cut_value = (total_edges - E_0) / 2
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    # Track minimum gap across ALL excited states (not degenerate with ground)
    k_index = degeneracy_s1
    min_gap = np.inf
    s_at_min = 0.0
    excited_level_at_min = k_index
    
    for s in s_points:
        H_s = get_aqc_hamiltonian(s, H_B, H_P)
        
        # Compute ALL eigenvalues for maximum accuracy
        eigenvalues = eigh(H_s, eigvals_only=True)
        
        if verbose:
            percent_complete = int((s / s_points[-1]) * 100)
            print(f"\r   Current graph progress: {percent_complete}% complete", end="")
        
        # Find minimum gap among all non-degenerate excited states
        # This checks ALL eigenvalues starting from k_index onwards
        gaps_at_s = eigenvalues[k_index:] - eigenvalues[0]
        min_gap_at_s_idx = np.argmin(gaps_at_s)
        gap_at_s = gaps_at_s[min_gap_at_s_idx]
        
        if gap_at_s < min_gap:
            min_gap = gap_at_s
            s_at_min = s
            excited_level_at_min = k_index + min_gap_at_s_idx
    
    return float(min_gap), float(s_at_min), int(degeneracy_s1), max_cut_value, int(excited_level_at_min)


def analyze_spectrum_for_visualization(
    H_B: np.ndarray,
    H_P: np.ndarray,
    s_points: np.ndarray,
    num_edges: int,
    max_degeneracy_check: int = 40
) -> dict:
    """
    Comprehensive spectrum analysis for visualization purposes.
    
    Computes full eigenvalue evolution and identifies minimum gap with
    proper degeneracy handling.
    
    Args:
        H_B: Initial Hamiltonian matrix
        H_P: Problem Hamiltonian matrix
        s_points: Array of s values to sample
        num_edges: Number of edges in the graph
        max_degeneracy_check: Maximum number of eigenvalues to check for degeneracy
                              (default 100 should cover most cases)
        
    Returns:
        Dictionary containing:
        - 'all_eigenvalues': Full spectrum evolution array
        - 'min_gap': Minimum spectral gap value
        - 's_at_min': s value where minimum occurs
        - 'min_gap_idx': Index in s_points where minimum occurs
        - 'degeneracy': Ground state degeneracy at s=1
        - 'level1': Ground state index (always 0)
        - 'level2': Excited state index that creates minimum gap
        - 'max_cut_value': Maximum cut value
    """
    # Get full spectrum evolution
    all_eigenvalues = compute_spectrum_evolution(H_B, H_P, s_points)
    
    # Get degeneracy at s=1 - check enough eigenvalues to detect high degeneracy
    k_vals_to_check = min(max_degeneracy_check, H_P.shape[0])
    evals_final = eigh(H_P, eigvals_only=True, subset_by_index=(0, k_vals_to_check - 1))
    _, degeneracy = find_first_gap(evals_final)
    
    # Calculate max cut value
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    # Find minimum gap (ignoring degenerate states)
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

# =========================================================================
# GRAPH FILE I/O
# =========================================================================

def extract_graph_params(filename: str) -> Tuple[int, int, int]:
    """
    Extract graph parameters from GENREG filename format.
    
    GENREG filenames follow the pattern: N_k_g.ext
    where N = number of vertices, k = degree, g = girth
    
    Args:
        filename: Graph filename (e.g., "12_3_3.asc" or "path/to/10_3_3.scd")
        
    Returns:
        Tuple of (n, k, girth) where n is vertices, k is degree, girth is minimum cycle length
        
    Examples:
        >>> extract_graph_params("graphs_rawdata/12_3_3.scd")
        (12, 3, 3)
        >>> extract_graph_params("/path/to/10_3_3.asc")
        (10, 3, 3)
    """
    import os
    basename = os.path.basename(filename)
    # Remove extension
    name_without_ext = basename.rsplit('.', 1)[0]
    # Split by underscore
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
    
    Args:
        adj_dict: Dictionary mapping vertex -> list of neighbors (1-indexed)
        
    Returns:
        List of edges as (v1, v2) tuples (0-indexed), with v1 < v2
    """
    edges = []
    seen = set()
    
    for v, neighbors in adj_dict.items():
        for n in neighbors:
            edge = (min(v, n), max(v, n))
            if edge not in seen:
                seen.add(edge)
                # Convert to 0-indexed
                edges.append((edge[0] - 1, edge[1] - 1))
    return sorted(edges)


def parse_asc_file(filename: str) -> List[List[Tuple[int, int]]]:
    """
    Parse GENREG .asc file and return list of graphs as 0-indexed edge lists.
    
    The .asc format contains human-readable adjacency lists with additional
    metadata (Taillenweite = girth, Ordnung = automorphism group order).
    
    Args:
        filename: Path to .asc file
        
    Returns:
        List of graphs, where each graph is a list of (v1, v2) edge tuples (0-indexed)
        
    Example:
        >>> graphs = parse_asc_file("graphs_rawdata/12_3_3.asc")
        >>> len(graphs)  # Number of graphs in file
        85
        >>> graphs[0]  # First graph's edge list
        [(0, 1), (0, 2), (0, 3), ...]
    """
    graphs = []
    current_graph = {}
    in_adjacency = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('Graph'):
                # Starting a new graph
                if current_graph:
                    # Save previous graph
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
    
    # Don't forget the last graph
    if current_graph:
        graphs.append(adjacency_to_edges(current_graph))
    
    return graphs


def shortcode_to_edges(code: List[int], n: int, k: int) -> List[Tuple[int, int]]:
    """
    Convert GENREG shortcode representation to edge list.
    
    The shortcode format lists, for each vertex v in order, only the neighbors
    w where w > v. This avoids duplicate edge entries.
    
    Args:
        code: Shortcode array (1-indexed vertex numbers)
        n: Number of vertices in graph
        k: Degree of each vertex (k-regular graph)
        
    Returns:
        List of edges as (v1, v2) tuples (0-indexed), sorted
        
    Example:
        >>> # Complete graph K4: edges (1-2), (1-3), (1-4), (2-3), (2-4), (3-4)
        >>> shortcode_to_edges([2, 3, 4, 3, 4, 4], 4, 3)
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    """
    edges = []
    degree_count = [0] * n  # Track degree of each vertex
    v = 0  # Current vertex (0-indexed)
    
    for w_1indexed in code:
        w = int(w_1indexed) - 1  # Convert to 0-indexed (ensure plain int)
        
        # Find the current vertex (one that hasn't reached degree k)
        while v < n and degree_count[v] == k:
            v += 1
        
        if v >= n:
            # Should not happen with valid data
            break
        
        # Create edge (v, w) with plain Python ints
        edges.append((int(v), int(w)))
        degree_count[v] += 1
        degree_count[w] += 1
    
    return sorted(edges)


def parse_scd_file(filename: str, n: Optional[int] = None, k: Optional[int] = None) -> List[List[Tuple[int, int]]]:
    """
    Parse GENREG .scd (shortcode) binary file and return list of graphs.
    
    The .scd format uses differential compression:
    - First value: number of common prefix elements with previous code
    - Remaining values: new/different elements of the code
    - First graph always starts with 0 (no common prefix)
    
    Args:
        filename: Path to .scd file
        n: Number of vertices (auto-detected from filename if None)
        k: Vertex degree (auto-detected from filename if None)
        
    Returns:
        List of graphs, where each graph is a list of (v1, v2) edge tuples (0-indexed)
        
    Example:
        >>> graphs = parse_scd_file("graphs_rawdata/12_3_3.scd")
        >>> len(graphs)  # Number of graphs
        85
        >>> graphs[0]  # First graph's edge list
        [(0, 1), (0, 2), (0, 3), ...]
    """
    # Auto-detect parameters from filename if not provided
    if n is None or k is None:
        n_auto, k_auto, _ = extract_graph_params(filename)
        n = n if n is not None else n_auto
        k = k if k is not None else k_auto
    
    num_edges = n * k // 2
    
    # Read binary file
    with open(filename, 'rb') as f:
        values = np.fromfile(f, dtype=np.uint8)
    
    graphs = []
    read_pos = 0
    code = []
    
    while read_pos < len(values):
        # Read common prefix length
        samebits = int(values[read_pos])
        read_pos += 1
        
        # Calculate how many new values to read
        readbits = num_edges - samebits
        
        # Update code: keep first 'samebits' elements, append new values
        code = code[:samebits] + list(values[read_pos:read_pos + readbits])
        read_pos += readbits
        
        # Convert shortcode to edge list
        edges = shortcode_to_edges(code, n, k)
        graphs.append(edges)
    
    return graphs


def load_graphs_from_file(filename: str) -> List[List[Tuple[int, int]]]:
    """
    Load graphs from GENREG file (supports both .asc and .scd formats).
    
    Automatically detects file format based on extension and calls the
    appropriate parser. This is the recommended way to load graphs as it
    provides format-agnostic loading.
    
    Args:
        filename: Path to .asc or .scd file
        
    Returns:
        List of graphs, where each graph is a list of (v1, v2) edge tuples (0-indexed)
        
    Raises:
        ValueError: If file extension is not .asc or .scd
        
    Example:
        >>> graphs = load_graphs_from_file("graphs_rawdata/12_3_3.scd")  # Loads SCD
        >>> graphs = load_graphs_from_file("graphs_rawdata/10_3_3.asc")  # Loads ASC
        >>> len(graphs)
        85
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

