#!/usr/bin/env python3
"""
Verification script: Creates a specific 3-regular graph and shows 
the Hamiltonian construction in detail to verify correctness.
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Small example with N=6
N = 6

# Define Pauli matrices
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

print("="*70)
print("  HAMILTONIAN VERIFICATION FOR N=6")
print("="*70)

# Create a specific 3-regular graph
G = nx.random_regular_graph(d=3, n=N, seed=42)  # Use seed for reproducibility
edges = list(G.edges())

print(f"\nGraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"Edges: {edges}")
print(f"\nDegree sequence: {dict(G.degree())}")

# Build Hamiltonians
print(f"\n{'='*70}")
print("Building Hamiltonians...")
print(f"{'='*70}")

H_B = build_H_initial(N)
H_P = build_H_problem(N, edges)

print(f"\n✓ H_initial shape: {H_B.shape} (2^{N} = {2**N})")
print(f"✓ H_problem shape: {H_P.shape}")

# Properties of H_initial
print(f"\nH_initial = -∑ᵢ X̂ᵢ properties:")
print(f"  - Is Hermitian: {np.allclose(H_B, H_B.conj().T)}")
print(f"  - Trace: {np.trace(H_B).real:.6f}")
print(f"  - Is real: {np.allclose(H_B.imag, 0)}")
print(f"  - Min element: {H_B.real.min():.4f}")
print(f"  - Max element: {H_B.real.max():.4f}")

# Properties of H_problem
print(f"\nH_problem = ∑₍ᵢ,ⱼ₎ ẐᵢẐⱼ properties:")
print(f"  - Is Hermitian: {np.allclose(H_P, H_P.conj().T)}")
print(f"  - Trace: {np.trace(H_P).real:.6f}")
print(f"  - Is diagonal: {np.allclose(H_P, np.diag(np.diag(H_P)))}")
print(f"  - Is real: {np.allclose(H_P.imag, 0)}")

# Test at different s values
print(f"\n{'='*70}")
print("Spectral gap at different s values:")
print(f"{'='*70}")

s_values = [0.0, 0.25, 0.5, 0.75, 1.0]
gaps = []

for s in s_values:
    H_s = (1 - s) * H_B + s * H_P
    evals = eigh(H_s, eigvals_only=True, subset_by_index=(0, 1))
    gap = evals[1] - evals[0]
    gaps.append(gap)
    print(f"  s={s:.2f}: E₀={evals[0]:8.4f}, E₁={evals[1]:8.4f}, Δ={gap:8.4f}")

# Find minimum with higher resolution
print(f"\n{'='*70}")
print("Finding minimum gap with high resolution...")
print(f"{'='*70}")

s_points = np.linspace(0, 1, 200)
min_gap = np.inf
s_at_min = 0
H_at_min = None

# Store eigenvalues for plotting
E0_array = []
E1_array = []
gap_array = []

for s in s_points:
    H_s = (1 - s) * H_B + s * H_P
    evals = eigh(H_s, eigvals_only=True, subset_by_index=(0, 1))
    
    E0_array.append(evals[0])
    E1_array.append(evals[1])
    gap = evals[1] - evals[0]
    gap_array.append(gap)
    
    if gap < min_gap:
        min_gap = gap
        s_at_min = s
        H_at_min = H_s.copy()
        E0_min = evals[0]
        E1_min = evals[1]

E0_array = np.array(E0_array)
E1_array = np.array(E1_array)
gap_array = np.array(gap_array)

print(f"\n✅ Minimum gap found:")
print(f"   Δ_min = {min_gap:.6f}")
print(f"   at s = {s_at_min:.6f}")
print(f"   E₀ = {E0_min:.6f}")
print(f"   E₁ = {E1_min:.6f}")

# Verify by full diagonalization
all_evals = np.linalg.eigvalsh(H_at_min)
print(f"\n   Full diagonalization verification:")
print(f"   Gap = {all_evals[1] - all_evals[0]:.6f}")
print(f"   Match: {np.isclose(all_evals[1] - all_evals[0], min_gap)}")

# Show spectrum
print(f"\n   Energy spectrum (lowest 10 levels):")
for i, E in enumerate(all_evals[:10]):
    marker = "◄" if i < 2 else ""
    print(f"   E_{i}: {E:10.6f} {marker}")

# Save the Hamiltonian
filename = f"H_min_N{N}_detailed.npy"
np.save(filename, H_at_min)
print(f"\n💾 Saved H(s_min) to: {filename}")

# Plot the eigenvalues vs s
print(f"\n{'='*70}")
print("Plotting eigenvalue evolution...")
print(f"{'='*70}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top plot: Both eigenvalues
ax1.plot(s_points, E0_array, 'b-', linewidth=2, label='E₀ (ground state)')
ax1.plot(s_points, E1_array, 'r-', linewidth=2, label='E₁ (first excited)')
ax1.axvline(s_at_min, color='green', linestyle='--', alpha=0.7, 
            label=f'Min gap at s={s_at_min:.3f}')
ax1.set_xlabel('s (interpolation parameter)', fontsize=12)
ax1.set_ylabel('Energy', fontsize=12)
ax1.set_title(f'Eigenvalue Evolution for N={N} (3-regular graph)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Bottom plot: Spectral gap
ax2.plot(s_points, gap_array, 'purple', linewidth=2, label='Δ(s) = E₁ - E₀')
ax2.axhline(min_gap, color='orange', linestyle='--', alpha=0.7, 
            label=f'Δ_min = {min_gap:.6f}')
ax2.axvline(s_at_min, color='green', linestyle='--', alpha=0.7)
ax2.fill_between(s_points, 0, gap_array, alpha=0.2, color='purple')
ax2.set_xlabel('s (interpolation parameter)', fontsize=12)
ax2.set_ylabel('Spectral Gap Δ(s)', fontsize=12)
ax2.set_title('Spectral Gap Evolution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plot_filename = f"eigenvalues_vs_s_N{N}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"\n📊 Plot saved to: {plot_filename}")
plt.show()

# Additional verification: Check that H_problem is diagonal in computational basis
print(f"\n{'='*70}")
print("Additional Checks:")
print(f"{'='*70}")

# For H_problem at s=1, verify it's diagonal
H_final = H_P
print(f"\nH_problem (s=1) diagonal elements (first 10):")
diag = np.diag(H_final.real)
for i in range(min(10, len(diag))):
    bitstring = format(i, f'0{N}b')
    print(f"  |{bitstring}⟩: {diag[i]:6.1f}")

# Count degeneracies
print(f"\nEnergy level degeneracies at s=1:")
unique, counts = np.unique(all_evals.round(6), return_counts=True)
for e, count in zip(unique[:10], counts[:10]):
    print(f"  E = {e:8.4f}: {count:3d} states")

print(f"\n{'='*70}")
print("Verification complete!")
print(f"{'='*70}")
