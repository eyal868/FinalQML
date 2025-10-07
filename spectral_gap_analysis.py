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

N_QUBITS = 10          # Maximum number of qubits (nodes in the graph)
NUM_GRAPHS = 200       # Number of random 3-regular graph instances
S_RESOLUTION = 200     # Number of points to sample along s ∈ [0, 1]

# Output filename
OUTPUT_FILENAME = f'outputs/Delta_min_3_regular_N{N_QUBITS}_{NUM_GRAPHS}graphs.csv'

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
# 4. SPECTRAL GAP CALCULATION (FIXED FOR DEGENERACY)
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
                              s_points: np.ndarray) -> Tuple[float, float, int]:
    """
    Calculates minimum spectral gap handling ground state degeneracy properly.
    
    CORRECT METHOD:
    1. First determine degeneracy at s=1 (pure problem Hamiltonian)
    2. Find which eigenvalue index k is the first non-degenerate one
    3. Track gap E_k - E_0 throughout the entire evolution
    
    This ensures we're tracking the SAME eigenvalue level throughout,
    not switching between E_1, E_2, etc. as degeneracies appear.
    
    Returns:
        Tuple of (min_gap, s_at_min_gap, degeneracy_at_s1)
    """
    # Step 1: Determine degeneracy at s=1 (problem Hamiltonian only)
    H_final = H_P  # At s=1, H(1) = H_problem
    k_vals = min(20, H_final.shape[0])  # Check more eigenvalues for safety
    evals_final = eigh(H_final, eigvals_only=True, subset_by_index=(0, k_vals-1))
    _, degeneracy_s1 = find_first_gap(evals_final)
    
    # Step 2: Track E_k - E_0 where k is the degeneracy at s=1
    # This is the eigenvalue INDEX we need to track throughout
    k_index = degeneracy_s1  # If ground state is deg_s1-fold, track E_{deg_s1}
    
    min_gap = np.inf
    s_at_min = 0.0
    
    for s in s_points:
        H_s = get_aqc_hamiltonian(s, H_B, H_P)
        
        # Get enough eigenvalues to track E_k
        # Use larger buffer to handle cases where degeneracy increases during evolution
        k_vals_needed = min(k_index + 10, H_s.shape[0])  # Larger safety buffer
        eigenvalues = eigh(H_s, eigvals_only=True, subset_by_index=(0, k_vals_needed-1))
        
        # Gap is E_k - E_0 where k is determined by s=1 degeneracy
        if k_index < len(eigenvalues):
            gap = eigenvalues[k_index] - eigenvalues[0]
        else:
            # Fallback: compute more eigenvalues if needed
            print(f"Warning: Need more eigenvalues at s={s:.3f}, k_index={k_index}, computed={len(eigenvalues)}")
            eigenvalues_all = eigh(H_s, eigvals_only=True, subset_by_index=(0, min(k_index+1, H_s.shape[0]-1)))
            gap = eigenvalues_all[k_index] - eigenvalues_all[0]
        
        if gap < min_gap:
            min_gap = gap
            s_at_min = s
    
    return float(min_gap), float(s_at_min), int(degeneracy_s1)

# =========================================================================
# 5. MAIN EXECUTION
# =========================================================================

def main():
    """Main execution function with robust gap calculation."""
    print("=" * 70)
    print("  AQC SPECTRAL GAP ANALYSIS FOR 3-REGULAR GRAPHS")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  • N_QUBITS: {N_QUBITS}")
    print(f"  • NUM_GRAPHS: {NUM_GRAPHS}")
    print(f"  • S_RESOLUTION: {S_RESOLUTION}")
    print(f"  • Degeneracy tolerance: {DEGENERACY_TOL}")
    print(f"  • Hilbert space dimension: 2^{N_QUBITS} = {2**N_QUBITS}")
    
    # Validate configuration
    if N_QUBITS % 2 != 0:
        print(f"\n⚠️  WARNING: N={N_QUBITS} is odd. 3-regular graphs require N to be even.")
    
    # Initialize
    s_points = np.linspace(0.0, 1.0, S_RESOLUTION)
    data = []
    
    # Pre-calculate the Initial Hamiltonian (H_B) - same for all graphs
    print(f"\n🔨 Building H_initial (Mixer) matrix of size {2**N_QUBITS}×{2**N_QUBITS}...")
    H_B = build_H_initial(N_QUBITS)
    print("   ✓ Done")
    
    # Start timer
    start_time = time.time()
    print(f"\n🚀 Starting analysis of {NUM_GRAPHS} random 3-regular graphs...")
    print("-" * 70)
    
    degeneracy_counts = []
    
    for i in range(NUM_GRAPHS):
        # Generate a random 3-regular graph (degree d=3 for all nodes)
        try:
            G = nx.random_regular_graph(d=3, n=N_QUBITS)
        except nx.NetworkXError as e:
            print(f"❌ Error generating graph {i+1}: {e}")
            continue
        
        edges = list(G.edges())
        
        # Build the Problem Hamiltonian (H_P) for this specific graph
        H_P = build_H_problem(N_QUBITS, edges)
        
        # Calculate the Minimum Spectral Gap (handling degeneracy)
        delta_min, s_min, max_deg = calculate_min_gap_robust(H_B, H_P, s_points)
        
        degeneracy_counts.append(max_deg)
        
        # Store results
        data.append({
            'N': N_QUBITS,
            'Graph_ID': i + 1,
            'Delta_min': delta_min,
            's_at_min': s_min,
            'Max_degeneracy': max_deg,
            'Edges': str(edges)
        })
        
        # Progress reporting
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (NUM_GRAPHS - i - 1)
            print(f"  [{i+1:3d}/{NUM_GRAPHS}] Δ_min={delta_min:.6f} at s={s_min:.3f} "
                  f"(deg={max_deg}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
    # Save results to CSV
    print("-" * 70)
    print(f"\n💾 Saving results to {OUTPUT_FILENAME}...")
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILENAME, index=False)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   • Total graphs processed: {len(data)}")
    print(f"   • Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   • Average time per graph: {total_time/len(data):.2f} seconds")
    
    print(f"\n📈 Spectral Gap Statistics:")
    print(f"   • Mean Δ_min: {df['Delta_min'].mean():.6f}")
    print(f"   • Std Δ_min: {df['Delta_min'].std():.6f}")
    print(f"   • Min Δ_min: {df['Delta_min'].min():.6f} (hardest instance)")
    print(f"   • Max Δ_min: {df['Delta_min'].max():.6f} (easiest instance)")
    print(f"   • Median Δ_min: {df['Delta_min'].median():.6f}")
    
    print(f"\n📈 Degeneracy Statistics:")
    print(f"   • Mean max degeneracy: {df['Max_degeneracy'].mean():.2f}")
    print(f"   • Max degeneracy found: {df['Max_degeneracy'].max()}")
    print(f"   • Graphs with degeneracy > 1: {(df['Max_degeneracy'] > 1).sum()}")
    
    print(f"\n📁 Data saved to: {OUTPUT_FILENAME}")
    print("=" * 70)


if __name__ == "__main__":
    main()
