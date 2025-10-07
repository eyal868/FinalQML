#!/usr/bin/env python3
"""
=========================================================================
Spectral Gap Analysis for Random 3-Regular Graphs
=========================================================================
Research Goal: Analyze the minimum energy gap (Δ_min) of the Adiabatic 
Quantum Computing (AQC) Hamiltonian H(s) for Max-Cut on 3-regular graphs.

This data will be used to study the connection between Δ_min and QAOA 
performance, where AQC runtime scales as T ∝ 1/(Δ_min)².

METHODOLOGY: Handles ground state degeneracy correctly by:
1. Determining degeneracy at s=1 (problem Hamiltonian)
2. Tracking E_k - E_0 throughout evolution (k = degeneracy)
3. Finding minimum of this gap across all s
=========================================================================
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
import pandas as pd
import time
from typing import List, Tuple

# =========================================================================
# 1. CONFIGURATION PARAMETERS
# =========================================================================

# GENREG files containing complete enumeration of 3-regular graphs
GENREG_FILES = {
    10: '10_3_3.asc',  # 19 graphs
    12: '12_3_3.asc'   # 85 graphs
}

S_RESOLUTION = 200             # Number of points to sample along s ∈ [0, 1]

# Output filename
OUTPUT_FILENAME = 'outputs/Delta_min_3_regular_N10-12_ALL_GENREG_graphs.csv'

# Tolerance for considering eigenvalues as degenerate
DEGENERACY_TOL = 1e-8

# =========================================================================
# 2. PAULI MATRICES DEFINITION
# =========================================================================

# Single-qubit Pauli matrices
SIGMA_X = np.array([[0, 1], 
                     [1, 0]], dtype=complex)

SIGMA_Z = np.array([[1, 0], 
                     [0, -1]], dtype=complex)

IDENTITY = np.eye(2, dtype=complex)

# =========================================================================
# 3. HAMILTONIAN CONSTRUCTION FUNCTIONS
# =========================================================================

def pauli_tensor_product(op_list: List[np.ndarray]) -> np.ndarray:
    """Computes the tensor product of a list of 2x2 matrices."""
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result


def get_pauli_term(N: int, pauli_type: str, index1: int, index2: int = -1) -> np.ndarray:
    """Generates an N-qubit operator (X_i or Z_i⊗Z_j) as a 2^N × 2^N matrix."""
    operators = [IDENTITY] * N
    
    if pauli_type == 'X':
        operators[index1] = SIGMA_X
    elif pauli_type == 'ZZ':
        operators[index1] = SIGMA_Z
        operators[index2] = SIGMA_Z
    
    return pauli_tensor_product(operators)


def build_H_initial(N: int) -> np.ndarray:
    """Builds the Initial (Mixer) Hamiltonian: H_initial = -∑ᵢ X̂ᵢ"""
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        H_B += get_pauli_term(N, 'X', i)
    return -H_B


def build_H_problem(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Builds the Problem (Cost) Hamiltonian: H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ"""
    H_P = np.zeros((2**N, 2**N), dtype=complex)
    for u, v in edges:
        H_P += get_pauli_term(N, 'ZZ', u, v)
    return H_P


def get_aqc_hamiltonian(s: float, H_B: np.ndarray, H_P: np.ndarray) -> np.ndarray:
    """Returns H(s) = (1-s)·H_initial + s·H_problem"""
    return (1 - s) * H_B + s * H_P

# =========================================================================
# 4. GENREG FILE PARSING
# =========================================================================

def parse_asc_file(filename: str) -> List[List[Tuple[int, int]]]:
    """
    Parse GENREG .asc file and return list of graphs as edge lists.
    
    Args:
        filename: Path to .asc file
        
    Returns:
        List of edge lists, where each edge list is a list of (vertex1, vertex2) tuples
        Vertices are 0-indexed.
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
                # End of adjacency list for this graph
                in_adjacency = False
                
            elif in_adjacency and ':' in line:
                # Parse adjacency line: "1 : 2 3 4"
                try:
                    parts = line.split(':')
                    vertex = int(parts[0].strip())
                    neighbors = [int(x) for x in parts[1].split()]
                    current_graph[vertex] = neighbors
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue
    
    # Don't forget last graph
    if current_graph:
        graphs.append(adjacency_to_edges(current_graph))
    
    return graphs


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
            # Create canonical edge representation (smaller vertex first)
            edge = (min(v, n), max(v, n))
            if edge not in seen:
                seen.add(edge)
                # Convert to 0-indexed
                edges.append((edge[0] - 1, edge[1] - 1))
    
    return sorted(edges)

# =========================================================================
# 5. SPECTRAL GAP CALCULATION (FIXED FOR DEGENERACY)
# =========================================================================

def find_first_gap(eigenvalues: np.ndarray, tol: float = DEGENERACY_TOL) -> Tuple[float, int]:
    """
    Find the gap between ground state and first non-degenerate excited state.
    
    Args:
        eigenvalues: Sorted array of eigenvalues (ascending)
        tol: Tolerance for considering eigenvalues as degenerate
        
    Returns:
        Tuple of (gap, degeneracy) where:
        - gap: Energy difference to first non-degenerate excited state
        - degeneracy: Number of degenerate ground states
    """
    E0 = eigenvalues[0]
    
    # Find all eigenvalues degenerate with ground state
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


def calculate_min_gap_robust(H_B: np.ndarray, H_P: np.ndarray, 
                              s_points: np.ndarray, num_edges: int) -> Tuple[float, float, int, int]:
    """
    Calculates minimum spectral gap handling ground state degeneracy properly.
    
    CORRECT METHOD:
    1. First determine degeneracy at s=1 (pure problem Hamiltonian)
    2. Find which eigenvalue index k is the first non-degenerate one
    3. Track gap E_k - E_0 throughout the entire evolution
    
    This ensures we're tracking the SAME eigenvalue level throughout,
    not switching between E_1, E_2, etc. as degeneracies appear.
    
    Returns:
        Tuple of (min_gap, s_at_min_gap, degeneracy_at_s1, max_cut_value)
    """
    # Step 1: Determine degeneracy at s=1 (problem Hamiltonian only)
    H_final = H_P  # At s=1, H(1) = H_problem
    k_vals = min(10, H_final.shape[0])
    evals_final = eigh(H_final, eigvals_only=True, subset_by_index=(0, k_vals-1))
    _, degeneracy_s1 = find_first_gap(evals_final)
    
    # Calculate max cut value from ground state energy
    # For H_problem = ∑_{(i,j)∈E} Z_i Z_j:
    # - Edges in cut contribute -1, edges not in cut contribute +1
    # - Cut_value = (total_edges - E_0) / 2
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    # Step 2: Track E_k - E_0 where k is the degeneracy at s=1
    # This is the eigenvalue INDEX we need to track throughout
    k_index = degeneracy_s1  # If ground state is deg_s1-fold, track E_{deg_s1}
    
    min_gap = np.inf
    s_at_min = 0.0
    
    for s in s_points:
        H_s = get_aqc_hamiltonian(s, H_B, H_P)
        
        # Get enough eigenvalues to track E_k
        k_vals_needed = min(k_index + 3, H_s.shape[0])  # Small buffer for safety
        eigenvalues = eigh(H_s, eigvals_only=True, subset_by_index=(0, k_vals_needed-1))
        
        # Gap is E_k - E_0 where k is determined by s=1 degeneracy
        if k_index < len(eigenvalues):
            gap = eigenvalues[k_index] - eigenvalues[0]
        else:
            gap = eigenvalues[-1] - eigenvalues[0]  # Fallback
        
        if gap < min_gap:
            min_gap = gap
            s_at_min = s
    
    return float(min_gap), float(s_at_min), int(degeneracy_s1), max_cut_value

# =========================================================================
# 6. MAIN EXECUTION
# =========================================================================

def main():
    """Main execution function using complete GENREG graph enumeration."""
    print("=" * 70)
    print("  AQC SPECTRAL GAP ANALYSIS FOR 3-REGULAR GRAPHS")
    print("  Using Complete GENREG Graph Enumeration")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  • S_RESOLUTION: {S_RESOLUTION}")
    print(f"  • Degeneracy tolerance: {DEGENERACY_TOL}")
    print(f"  • Source: Complete enumeration from GENREG")
    print(f"\n📁 GENREG Files:")
    for N, filename in GENREG_FILES.items():
        print(f"  • N={N}: {filename}")
    
    # Initialize
    s_points = np.linspace(0.0, 1.0, S_RESOLUTION)
    data = []
    
    # Start timer
    start_time = time.time()
    print(f"\n🚀 Starting analysis...")
    print("-" * 70)
    
    graph_counter = 0  # Global graph counter for unique IDs
    
    # Loop over GENREG files
    for N, filename in sorted(GENREG_FILES.items()):
        print(f"\n▶ Processing N={N} (Hilbert space dimension: 2^{N} = {2**N})")
        print(f"  📖 Reading graphs from {filename}...", end=" ")
        
        # Parse all graphs from file
        try:
            graphs = parse_asc_file(filename)
            print(f"✓ Found {len(graphs)} graphs")
        except FileNotFoundError:
            print(f"\n  ❌ Error: File not found: {filename}")
            continue
        except Exception as e:
            print(f"\n  ❌ Error parsing file: {e}")
            continue
        
        # Pre-calculate the Initial Hamiltonian (H_B) for this N
        print(f"  🔨 Building H_initial matrix {2**N}×{2**N}...", end=" ")
        H_B = build_H_initial(N)
        print("✓")
        
        num_edges = 3 * N // 2  # For 3-regular graphs
        
        # Process each graph from GENREG
        for i, edges in enumerate(graphs):
            graph_counter += 1
            
            # Build the Problem Hamiltonian (H_P) for this specific graph
            H_P = build_H_problem(N, edges)
            
            # Calculate the Minimum Spectral Gap (handling degeneracy) and max cut
            delta_min, s_min, max_deg, max_cut = calculate_min_gap_robust(H_B, H_P, s_points, num_edges)
            
            # Store results
            data.append({
                'N': N,
                'Graph_ID': graph_counter,
                'Delta_min': delta_min,
                's_at_min': s_min,
                'Max_degeneracy': max_deg,
                'Max_cut_value': max_cut,
                'Edges': str(edges)
            })
            
            # Progress reporting
            report_freq = 5 if N <= 10 else 10
            if (i + 1) % report_freq == 0 or (i + 1) == len(graphs):
                elapsed = time.time() - start_time
                print(f"    [{i+1:3d}/{len(graphs)}] Δ_min={delta_min:.6f} s={s_min:.3f} "
                      f"cut={max_cut} deg={max_deg}")
    
    # Sort data by N first, then by Delta_min
    print("\n" + "-" * 70)
    print(f"\n📊 Sorting data by N and Delta_min...")
    df = pd.DataFrame(data)
    df = df.sort_values(['N', 'Delta_min'], ascending=[True, True])
    
    # Save results to CSV
    print(f"💾 Saving results to {OUTPUT_FILENAME}...")
    df.to_csv(OUTPUT_FILENAME, index=False)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   • Total graphs processed: {len(data)}")
    print(f"   • Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   • Average time per graph: {total_time/len(data):.2f} seconds")
    
    print(f"\n📈 Spectral Gap Statistics (Overall):")
    print(f"   • Mean Δ_min: {df['Delta_min'].mean():.6f}")
    print(f"   • Std Δ_min: {df['Delta_min'].std():.6f}")
    print(f"   • Min Δ_min: {df['Delta_min'].min():.6f} (hardest instance)")
    print(f"   • Max Δ_min: {df['Delta_min'].max():.6f} (easiest instance)")
    
    print(f"\n📈 Statistics by N:")
    for N in sorted(GENREG_FILES.keys()):
        df_n = df[df['N'] == N]
        if len(df_n) > 0:
            print(f"   N={N:2d}: Count={len(df_n):3d}, "
                  f"Δ_min={df_n['Delta_min'].mean():.6f}±{df_n['Delta_min'].std():.6f}, "
                  f"MaxCut={df_n['Max_cut_value'].mean():.1f}±{df_n['Max_cut_value'].std():.1f}")
    
    print(f"\n📁 Data saved to: {OUTPUT_FILENAME}")
    print("=" * 70)


if __name__ == "__main__":
    main()
