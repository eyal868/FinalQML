#!/usr/bin/env python3
"""
Quick test script to verify the spectral gap calculation works correctly
with a small example (N=4, 2 graphs).
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import time

# Small test configuration
N_QUBITS = 4
NUM_GRAPHS = 2
S_RESOLUTION = 50

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

def pauli_tensor_product(op_list):
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result

def get_pauli_term(N, pauli_type, index1, index2=-1):
    operators = [IDENTITY] * N
    if pauli_type == 'X':
        operators[index1] = SIGMA_X
    elif pauli_type == 'ZZ':
        operators[index1] = SIGMA_Z
        operators[index2] = SIGMA_Z
    return pauli_tensor_product(operators)

def build_H_initial(N):
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        H_B += get_pauli_term(N, 'X', i)
    return -H_B

def build_H_problem(N, edges):
    H_P = np.zeros((2**N, 2**N), dtype=complex)
    for u, v in edges:
        H_P += get_pauli_term(N, 'ZZ', u, v)
    return H_P

def get_aqc_hamiltonian(s, H_B, H_P):
    return (1 - s) * H_B + s * H_P

def calculate_min_gap(H_B, H_P, s_points):
    min_gap = np.inf
    s_at_min = 0.0
    H_at_min = None
    E0_at_min = None
    E1_at_min = None
    
    # Store eigenvalues for plotting
    E0_array = []
    E1_array = []
    
    for s in s_points:
        H_s = get_aqc_hamiltonian(s, H_B, H_P)
        eigenvalues = eigh(H_s, eigvals_only=True, subset_by_index=(0, 1))
        gap = eigenvalues[1] - eigenvalues[0]
        
        E0_array.append(eigenvalues[0])
        E1_array.append(eigenvalues[1])
        
        if gap < min_gap:
            min_gap = gap
            s_at_min = s
            H_at_min = H_s.copy()
            E0_at_min = eigenvalues[0]
            E1_at_min = eigenvalues[1]
    
    return (float(min_gap), float(s_at_min), H_at_min, E0_at_min, E1_at_min, 
            np.array(E0_array), np.array(E1_array))

# Run test
print("=" * 60)
print("  QUICK TEST: Spectral Gap Calculation")
print("=" * 60)
print(f"\nConfiguration: N={N_QUBITS}, {NUM_GRAPHS} graphs, {S_RESOLUTION} s-points")

s_points = np.linspace(0.0, 1.0, S_RESOLUTION)
H_B = build_H_initial(N_QUBITS)

print(f"\n✓ Built H_initial: {H_B.shape} matrix")
print(f"  H_initial is Hermitian: {np.allclose(H_B, H_B.conj().T)}")

start = time.time()

for i in range(NUM_GRAPHS):
    G = nx.random_regular_graph(d=3, n=N_QUBITS)
    edges = list(G.edges())
    H_P = build_H_problem(N_QUBITS, edges)
    delta_min, s_min, H_min, E0, E1, E0_array, E1_array = calculate_min_gap(H_B, H_P, s_points)
    
    print(f"\n{'='*60}")
    print(f"Graph {i+1}:")
    print(f"  Edges: {edges}")
    print(f"  Δ_min = {delta_min:.6f} at s = {s_min:.3f}")
    print(f"  E₀ = {E0:.6f}, E₁ = {E1:.6f}")
    print(f"  H_problem is Hermitian: {np.allclose(H_P, H_P.conj().T)}")
    
    # Display the Hamiltonian at minimum gap
    print(f"\n  H(s={s_min:.3f}) at minimum gap:")
    print(f"  Shape: {H_min.shape}")
    print(f"  Matrix (showing all elements for N=4):")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    print(H_min.real)  # Display real part (should be mostly real)
    
    # Verify properties
    print(f"\n  Matrix Properties:")
    print(f"    - Is Hermitian: {np.allclose(H_min, H_min.conj().T)}")
    print(f"    - Max imaginary part: {np.max(np.abs(H_min.imag)):.10f}")
    print(f"    - Trace: {np.trace(H_min):.6f}")
    
    # Verify eigenvalues by full diagonalization
    all_eigenvalues = np.linalg.eigvalsh(H_min)
    print(f"\n  Eigenvalue Verification (full diagonalization):")
    print(f"    - E₀ (ground): {all_eigenvalues[0]:.6f}")
    print(f"    - E₁ (1st excited): {all_eigenvalues[1]:.6f}")
    print(f"    - Gap from full diag: {all_eigenvalues[1] - all_eigenvalues[0]:.6f}")
    print(f"    - Matches calculated: {np.isclose(all_eigenvalues[1] - all_eigenvalues[0], delta_min)}")
    print(f"    - All eigenvalues: {all_eigenvalues[:5]}...")  # Show first 5

elapsed = time.time() - start
print(f"\n✅ Test completed in {elapsed:.2f} seconds")
print("=" * 60)

# Save the last Hamiltonian to file for detailed inspection
if NUM_GRAPHS > 0:
    filename = f"H_at_min_gap_N{N_QUBITS}_graph{NUM_GRAPHS}.txt"
    print(f"\n💾 Saving last H(s_min) matrix to: {filename}")
    with open(filename, 'w') as f:
        f.write(f"Hamiltonian H(s) at minimum spectral gap\n")
        f.write(f"="*70 + "\n")
        f.write(f"N_QUBITS: {N_QUBITS}\n")
        f.write(f"Graph ID: {NUM_GRAPHS}\n")
        f.write(f"Edges: {edges}\n")
        f.write(f"s at minimum: {s_min:.6f}\n")
        f.write(f"Δ_min: {delta_min:.6f}\n")
        f.write(f"E₀: {E0:.6f}\n")
        f.write(f"E₁: {E1:.6f}\n")
        f.write(f"\nH(s={s_min:.6f}) = (1-s)·H_initial + s·H_problem\n")
        f.write(f"="*70 + "\n\n")
        f.write("Matrix (16×16 for N=4):\n\n")
        np.savetxt(f, H_min.real, fmt='%8.4f')
        
    print(f"   ✓ Saved successfully")
    
    # Also save as numpy array for easy loading
    np_filename = f"H_at_min_gap_N{N_QUBITS}_graph{NUM_GRAPHS}.npy"
    np.save(np_filename, H_min)
    print(f"   ✓ Also saved as binary: {np_filename}")
    print(f"\n   To load in Python:")
    print(f"     H = np.load('{np_filename}')")
    
    # Plot eigenvalue evolution for the last graph
    print(f"\n{'='*60}")
    print("Creating eigenvalue evolution plot...")
    print(f"{'='*60}")
    
    gap_array = E1_array - E0_array
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top plot: Both eigenvalues
    ax1.plot(s_points, E0_array, 'b-', linewidth=2, label='E₀ (ground state)')
    ax1.plot(s_points, E1_array, 'r-', linewidth=2, label='E₁ (first excited)')
    ax1.axvline(s_min, color='green', linestyle='--', alpha=0.7, 
                label=f'Min gap at s={s_min:.3f}')
    ax1.set_xlabel('s (interpolation parameter)', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Eigenvalue Evolution for N={N_QUBITS} (Graph {NUM_GRAPHS})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    # Bottom plot: Spectral gap
    ax2.plot(s_points, gap_array, 'purple', linewidth=2, label='Δ(s) = E₁ - E₀')
    ax2.axhline(delta_min, color='orange', linestyle='--', alpha=0.7, 
                label=f'Δ_min = {delta_min:.6f}')
    ax2.axvline(s_min, color='green', linestyle='--', alpha=0.7)
    ax2.fill_between(s_points, 0, gap_array, alpha=0.2, color='purple')
    ax2.set_xlabel('s (interpolation parameter)', fontsize=12)
    ax2.set_ylabel('Spectral Gap Δ(s)', fontsize=12)
    ax2.set_title('Spectral Gap Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    ax2.set_ylim(bottom=-0.05*np.max(gap_array))  # Small negative space for visibility
    
    plt.tight_layout()
    plot_filename = f"eigenvalues_vs_s_N{N_QUBITS}_graph{NUM_GRAPHS}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {plot_filename}")
    plt.show()
    print("   ✓ Plot displayed")
