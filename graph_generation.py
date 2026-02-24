#!/usr/bin/env python3
"""
=========================================================================
Random Graph Generation Module
=========================================================================
Generates random graphs (Erdős-Rényi, random regular, Watts-Strogatz)
for spectral gap analysis and QAOA benchmarking.

All generators return edge lists compatible with the existing
Hamiltonian builders in aqc_spectral_utils.py.
=========================================================================
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict
import pandas as pd

from aqc_spectral_utils import (
    build_H_initial_sparse,
    build_H_problem_sparse,
    build_H_problem_sparse_weighted,
    find_min_gap_sparse,
    find_first_gap,
    DEGENERACY_TOL
)


# =========================================================================
# GRAPH GENERATORS
# =========================================================================

def generate_erdos_renyi(N: int, edge_prob: float, seed: int = None,
                         ensure_connected: bool = True,
                         max_retries: int = 1000) -> Optional[List[Tuple[int, int]]]:
    """
    Generate an Erdős-Rényi random graph G(N, edge_prob).

    Args:
        N: Number of vertices
        edge_prob: Probability of each edge existing
        seed: Random seed for reproducibility
        ensure_connected: If True, retry until connected graph found
        max_retries: Max attempts to find a connected graph

    Returns:
        Edge list as [(i, j), ...] with 0-based indexing, or None if failed
    """
    rng = np.random.RandomState(seed)
    for attempt in range(max_retries):
        s = rng.randint(0, 2**31) if attempt > 0 else (seed if seed is not None else rng.randint(0, 2**31))
        G = nx.erdos_renyi_graph(N, edge_prob, seed=s)
        if not ensure_connected or nx.is_connected(G):
            return sorted(list(G.edges()))
    return None


def generate_random_regular(N: int, degree: int, seed: int = None) -> Optional[List[Tuple[int, int]]]:
    """
    Generate a random regular graph with given degree.

    Args:
        N: Number of vertices (N * degree must be even)
        degree: Degree of each vertex
        seed: Random seed

    Returns:
        Edge list as [(i, j), ...] with 0-based indexing, or None if failed
    """
    try:
        G = nx.random_regular_graph(degree, N, seed=seed)
        return sorted(list(G.edges()))
    except nx.NetworkXError:
        return None


def generate_watts_strogatz(N: int, k: int, rewire_prob: float,
                            seed: int = None) -> List[Tuple[int, int]]:
    """
    Generate a Watts-Strogatz small-world graph.

    Args:
        N: Number of vertices
        k: Each node connected to k nearest neighbors in ring (must be even)
        rewire_prob: Probability of rewiring each edge
        seed: Random seed

    Returns:
        Edge list as [(i, j), ...]
    """
    G = nx.watts_strogatz_graph(N, k, rewire_prob, seed=seed)
    return sorted(list(G.edges()))


def generate_random_graphs_batch(
    N: int,
    num_graphs: int,
    model: str = "erdos_renyi",
    base_seed: int = 42,
    **params
) -> List[List[Tuple[int, int]]]:
    """
    Generate a batch of random graphs.

    Args:
        N: Number of vertices
        num_graphs: Number of graphs to generate
        model: One of "erdos_renyi", "random_regular", "watts_strogatz"
        base_seed: Base random seed (each graph gets base_seed + i)
        **params: Model-specific parameters:
            erdos_renyi: edge_prob (float)
            random_regular: degree (int)
            watts_strogatz: k (int), rewire_prob (float)

    Returns:
        List of edge lists
    """
    graphs = []
    for i in range(num_graphs):
        seed = base_seed + i
        if model == "erdos_renyi":
            edge_prob = params.get("edge_prob", 0.5)
            edges = generate_erdos_renyi(N, edge_prob, seed=seed)
        elif model == "random_regular":
            degree = params.get("degree", 3)
            edges = generate_random_regular(N, degree, seed=seed)
        elif model == "watts_strogatz":
            k = params.get("k", 4)
            rewire_prob = params.get("rewire_prob", 0.3)
            edges = generate_watts_strogatz(N, k, rewire_prob, seed=seed)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'erdos_renyi', 'random_regular', or 'watts_strogatz'")

        if edges is not None:
            graphs.append(edges)
        else:
            print(f"  Warning: Failed to generate graph {i} with model={model}, seed={seed}")

    return graphs


# =========================================================================
# SPECTRAL GAP COMPUTATION FOR RANDOM GRAPHS
# =========================================================================

def compute_spectral_gaps_for_graphs(
    N: int,
    graphs: List[List[Tuple[int, int]]],
    target_degeneracy: int = None,
    s_bounds: Tuple[float, float] = (0.01, 0.99),
    xtol: float = 1e-4,
    verbose: bool = True,
    graph_id_offset: int = 0,
    weights_list: Optional[List[List[float]]] = None
) -> pd.DataFrame:
    """
    Compute spectral gaps for a list of graphs.

    Outputs the same CSV schema as spectral_gap_analysis.py so all downstream
    tools (QAOA analysis, plotting, filtering) work unchanged.

    Args:
        N: Number of vertices/qubits
        graphs: List of edge lists
        target_degeneracy: If set, skip graphs not matching this degeneracy.
                          If None, accept all degeneracies.
        s_bounds: Optimization bounds for gap search
        xtol: Optimization tolerance
        verbose: Print progress
        graph_id_offset: Starting ID for graph numbering
        weights_list: Optional per-graph edge weights. If provided, computes
                      weighted spectral gaps.

    Returns:
        DataFrame with columns: N, Graph_ID, Delta_min, s_at_min,
        Max_degeneracy, Max_cut_value, Edges [, Weights]
    """
    from scipy.optimize import minimize_scalar
    from aqc_spectral_utils import get_gap_sparse, compute_weighted_optimal_cut

    H_B = build_H_initial_sparse(N)
    results = []

    for i, edges in enumerate(graphs):
        graph_id = graph_id_offset + i
        num_edges = len(edges)

        if verbose:
            print(f"  [{i+1}/{len(graphs)}] Graph {graph_id} ({num_edges} edges)...", end=" ")

        weights = weights_list[i] if weights_list is not None else None

        # Build problem Hamiltonian
        if weights is not None:
            H_P = build_H_problem_sparse_weighted(N, edges, weights)
        else:
            H_P = build_H_problem_sparse(N, edges)

        # Check degeneracy at s=1
        diagonal = H_P.diagonal()
        k_check = min(10, len(diagonal))
        evals_final = np.partition(diagonal, k_check - 1)[:k_check]
        evals_final = np.sort(evals_final)
        _, degeneracy = find_first_gap(evals_final, tol=DEGENERACY_TOL)

        if target_degeneracy is not None and degeneracy != target_degeneracy:
            if verbose:
                print(f"skipped (degeneracy={degeneracy})")
            continue

        # Compute max cut value
        E_0 = evals_final[0]
        if weights is not None:
            max_cut_value, _ = compute_weighted_optimal_cut(edges, weights, N)
        else:
            max_cut_value = int((num_edges - E_0) / 2)

        # Find minimum gap using Brent optimization with existing get_gap_sparse
        k_eig = degeneracy + 1

        def gap_at_s(s):
            return get_gap_sparse(s, H_B, H_P, k_eigenvalues=k_eig)

        result = minimize_scalar(gap_at_s, bounds=s_bounds, method='bounded',
                                 options={'xatol': xtol})
        min_gap = result.fun
        s_at_min = result.x

        if verbose:
            print(f"Δ={min_gap:.4f} at s={s_at_min:.3f} (deg={degeneracy})")

        row = {
            'N': N,
            'Graph_ID': graph_id,
            'Delta_min': min_gap,
            's_at_min': s_at_min,
            'Max_degeneracy': degeneracy,
            'Max_cut_value': max_cut_value,
            'Edges': str(edges),
        }
        if weights is not None:
            row['Weights'] = str(weights)

        results.append(row)

    return pd.DataFrame(results)
