#!/usr/bin/env python3
"""
Plot the full energy spectrum for the example from test_small_example.py
Highlighting E_0, E_1, and E_6 to show the methodology.
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Configuration - same as test_small_example.py
N = 4
S_RESOLUTION = 100

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

print("="*70)
print("  FULL SPECTRUM EVOLUTION - Example from test_small_example.py")
print("="*70)

# Generate the same graph as test_small_example.py
G = nx.random_regular_graph(d=3, n=N)
edges = list(G.edges())

print(f"\nGraph: N={N} nodes, {G.number_of_edges()} edges")
print(f"Edges: {edges}")

# Build Hamiltonians
H_B = build_H_initial(N)
H_P = build_H_problem(N, edges)

# Compute full spectrum at each s
print(f"\nComputing full spectrum for {S_RESOLUTION} values of s...")
s_points = np.linspace(0, 1, S_RESOLUTION)
all_eigenvalues = np.zeros((S_RESOLUTION, 2**N))

for i, s in enumerate(s_points):
    H_s = (1 - s) * H_B + s * H_P
    all_eigenvalues[i, :] = np.linalg.eigvalsh(H_s)
    
    if (i+1) % 25 == 0:
        print(f"  Progress: {i+1}/{S_RESOLUTION}")

print("  ✓ Done")

# Find minimum gap for E_6 - E_0
gaps = all_eigenvalues[:, 6] - all_eigenvalues[:, 0]
min_gap_idx = np.argmin(gaps)
min_gap = gaps[min_gap_idx]
s_at_min = s_points[min_gap_idx]

print(f"\nSpectral gap (E_6 - E_0):")
print(f"  Δ_min = {min_gap:.6f} at s = {s_at_min:.3f}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all eigenvalues in gray first
for i in range(2**N):
    if i not in [0, 1, 6]:  # Skip the ones we'll highlight
        ax.plot(s_points, all_eigenvalues[:, i], color='darkgray', linewidth=1.0, alpha=0.6)

# Highlight E_0, E_1, and E_6
ax.plot(s_points, all_eigenvalues[:, 0], 'b-', linewidth=3, alpha=0.9, label='E₀ (ground state)')
ax.plot(s_points, all_eigenvalues[:, 1], 'orange', linewidth=2.5, alpha=0.9, label='E₁')
ax.plot(s_points, all_eigenvalues[:, 6], 'r-', linewidth=3, alpha=0.9, label='E₆ (tracked for gap)')

# Mark minimum gap location
ax.axvline(s_at_min, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Min gap at s={s_at_min:.3f}')

ax.set_xlabel('s (interpolation parameter)', fontsize=14)
ax.set_ylabel('Energy', fontsize=14)
ax.set_title(f'Full Energy Spectrum Evolution (N={N}, {2**N} eigenvalues)', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')
ax.set_xlim(0, 1)

plt.tight_layout()
filename = f"outputs/example_full_spectrum_N{N}.png"
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f"\n📊 Plot saved to: {filename}")
plt.show()

print("\n" + "="*70)
print("✅ Visualization complete!")
print("="*70)
