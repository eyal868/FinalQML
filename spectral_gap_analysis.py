#!/usr/bin/env python3
"""
=========================================================================
Spectral Gap Analysis for Random 3-Regular Graphs
=========================================================================
Research Goal: Analyze the minimum energy gap (Δ_min) of the Adiabatic 
Quantum Computing (AQC) Hamiltonian H(s) for Max-Cut on 3-regular graphs.

This data will be used to study the connection between Δ_min and QAOA 
performance, where AQC runtime scales as T ∝ 1/(Δ_min)^2.
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
OUTPUT_FILENAME = f'Delta_min_3_regular_N{N_QUBITS}_{NUM_GRAPHS}graphs.csv'

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
    """
    Computes the tensor product of a list of 2x2 matrices.
    
    Args:
        op_list: List of 2x2 numpy arrays (Pauli operators or identity)
        
    Returns:
        The tensor product as a 2^N × 2^N matrix
    """
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result


def get_pauli_term(N: int, pauli_type: str, index1: int, index2: int = -1) -> np.ndarray:
    """
    Generates an N-qubit operator (X_i or Z_i⊗Z_j) as a 2^N × 2^N matrix.
    
    Args:
        N: Number of qubits
        pauli_type: 'X' for single-qubit X operator, 'ZZ' for two-qubit ZZ operator
        index1: First qubit index (0-indexed)
        index2: Second qubit index (used only for 'ZZ')
        
    Returns:
        The N-qubit operator as a matrix
    """
    operators = [IDENTITY] * N
    
    if pauli_type == 'X':
        # Transverse field term: X_index1
        operators[index1] = SIGMA_X
    elif pauli_type == 'ZZ':
        # Max-Cut term: Z_index1 ⊗ Z_index2
        operators[index1] = SIGMA_Z
        operators[index2] = SIGMA_Z
    
    return pauli_tensor_product(operators)


def build_H_initial(N: int) -> np.ndarray:
    """
    Builds the Initial (Mixer) Hamiltonian: H_initial = -∑ᵢ X̂ᵢ
    
    This is the transverse field Hamiltonian that drives transitions between
    computational basis states.
    
    Args:
        N: Number of qubits
        
    Returns:
        H_initial as a 2^N × 2^N complex matrix
    """
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        H_B += get_pauli_term(N, 'X', i)
    return -H_B


def build_H_problem(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Builds the Problem (Cost) Hamiltonian: H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ
    
    This is the Ising Max-Cut Hamiltonian. The ground state encodes the
    maximum cut of the graph.
    
    Args:
        N: Number of qubits
        edges: List of graph edges as (node_i, node_j) tuples
        
    Returns:
        H_problem as a 2^N × 2^N complex matrix
    """
    H_P = np.zeros((2**N, 2**N), dtype=complex)
    for u, v in edges:
        H_P += get_pauli_term(N, 'ZZ', u, v)
    return H_P


def get_aqc_hamiltonian(s: float, H_B: np.ndarray, H_P: np.ndarray) -> np.ndarray:
    """
    Returns the time-dependent AQC Hamiltonian:
    H(s) = (1-s)·H_initial + s·H_problem
    
    Args:
        s: Interpolation parameter ∈ [0, 1]
        H_B: Initial (mixer) Hamiltonian
        H_P: Problem (cost) Hamiltonian
        
    Returns:
        H(s) as a 2^N × 2^N complex matrix
    """
    return (1 - s) * H_B + s * H_P

# =========================================================================
# 4. SPECTRAL GAP CALCULATION
# =========================================================================

def calculate_min_gap(H_B: np.ndarray, H_P: np.ndarray, s_points: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the minimum spectral gap (Δ_min) along the adiabatic path.
    
    The spectral gap at each point is Δ(s) = E₁(s) - E₀(s), where E₀ and E₁
    are the ground state and first excited state energies, respectively.
    
    Args:
        H_B: Initial Hamiltonian
        H_P: Problem Hamiltonian
        s_points: Array of s values to sample
        
    Returns:
        Tuple of (min_gap, s_at_min_gap)
    """
    min_gap = np.inf
    s_at_min = 0.0
    
    for s in s_points:
        H_s = get_aqc_hamiltonian(s, H_B, H_P)
        
        # Use eigh with subset_by_index=(0, 1) for fast calculation 
        # of only the lowest 2 eigenvalues. This is crucial for performance.
        eigenvalues = eigh(H_s, eigvals_only=True, subset_by_index=(0, 1))
        
        E0 = eigenvalues[0]
        E1 = eigenvalues[1]
        gap = E1 - E0
        
        if gap < min_gap:
            min_gap = gap
            s_at_min = s
    
    return float(min_gap), float(s_at_min)

# =========================================================================
# 5. MAIN EXECUTION
# =========================================================================

def main():
    """
    Main execution function that:
    1. Generates random 3-regular graphs
    2. Constructs their Hamiltonians
    3. Calculates minimum spectral gaps
    4. Saves results to CSV
    """
    print("=" * 70)
    print("  AQC SPECTRAL GAP ANALYSIS FOR RANDOM 3-REGULAR GRAPHS")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  • N_QUBITS: {N_QUBITS}")
    print(f"  • NUM_GRAPHS: {NUM_GRAPHS}")
    print(f"  • S_RESOLUTION: {S_RESOLUTION}")
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
        
        # Calculate the Minimum Spectral Gap (Δ_min)
        delta_min, s_min = calculate_min_gap(H_B, H_P, s_points)
        
        # Store results
        data.append({
            'N': N_QUBITS,
            'Graph_ID': i + 1,
            'Delta_min': delta_min,
            's_at_min': s_min,
            'Edges': str(edges)
        })
        
        # Progress reporting
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (NUM_GRAPHS - i - 1)
            print(f"  [{i+1:3d}/{NUM_GRAPHS}] Δ_min={delta_min:.6f} at s={s_min:.3f} | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
    
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
    print(f"\n📈 Statistics:")
    print(f"   • Mean Δ_min: {df['Delta_min'].mean():.6f}")
    print(f"   • Std Δ_min: {df['Delta_min'].std():.6f}")
    print(f"   • Min Δ_min: {df['Delta_min'].min():.6f}")
    print(f"   • Max Δ_min: {df['Delta_min'].max():.6f}")
    print(f"\n📁 Data saved to: {OUTPUT_FILENAME}")
    print("=" * 70)


if __name__ == "__main__":
    main()
