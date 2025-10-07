#!/usr/bin/env python3
"""
Plot the full energy spectrum (all eigenvalues) vs s for a 3-regular graph.
This shows how ALL energy levels evolve during the adiabatic interpolation.
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Configuration - use smaller N for full spectrum calculation
N = 8  # 2^8 = 256 eigenvalues (manageable to plot)
S_RESOLUTION = 100  # Sample points along s

print("="*70)
print("  FULL ENERGY SPECTRUM vs s")
print("="*70)
print(f"\nConfiguration:")
print(f"  N = {N} qubits")
print(f"  Hilbert space dimension: 2^{N} = {2**N}")
print(f"  Number of eigenvalues: {2**N}")
print(f"  Resolution: {S_RESOLUTION} points")

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

# Generate a random 3-regular graph
print(f"\n🔨 Generating random 3-regular graph...")
G = nx.random_regular_graph(d=3, n=N, seed=42)
edges = list(G.edges())
print(f"   ✓ Graph has {G.number_of_edges()} edges")

# Build Hamiltonians
print(f"\n🔨 Building Hamiltonians...")
H_B = build_H_initial(N)
H_P = build_H_problem(N, edges)
print(f"   ✓ H_initial: {H_B.shape}")
print(f"   ✓ H_problem: {H_P.shape}")

# Compute full spectrum at each s
print(f"\n🔨 Computing full spectrum for {S_RESOLUTION} values of s...")
print("   (This may take a minute...)")

s_points = np.linspace(0, 1, S_RESOLUTION)
all_eigenvalues = np.zeros((S_RESOLUTION, 2**N))

import time
start = time.time()

for i, s in enumerate(s_points):
    H_s = (1 - s) * H_B + s * H_P
    # Compute ALL eigenvalues
    evals = np.linalg.eigvalsh(H_s)
    all_eigenvalues[i, :] = evals
    
    if (i+1) % 20 == 0:
        print(f"   Progress: {i+1}/{S_RESOLUTION} ({100*(i+1)/S_RESOLUTION:.0f}%)")

elapsed = time.time() - start
print(f"   ✓ Done in {elapsed:.2f} seconds")

# Find the minimum gap
gaps = all_eigenvalues[:, 1] - all_eigenvalues[:, 0]
min_gap_idx = np.argmin(gaps)
min_gap = gaps[min_gap_idx]
s_at_min = s_points[min_gap_idx]

print(f"\n📊 Spectral gap analysis:")
print(f"   Δ_min = {min_gap:.6f}")
print(f"   at s = {s_at_min:.3f}")

# Create comprehensive plots
print(f"\n🎨 Creating plots...")

fig = plt.figure(figsize=(14, 10))

# Main plot: Full spectrum
ax1 = plt.subplot(2, 2, (1, 2))

# Plot all eigenvalues as lines
for i in range(2**N):
    if i == 0:
        ax1.plot(s_points, all_eigenvalues[:, i], 'b-', linewidth=2, alpha=0.8, label='E₀ (ground)')
    elif i == 1:
        ax1.plot(s_points, all_eigenvalues[:, i], 'r-', linewidth=2, alpha=0.8, label='E₁ (1st excited)')
    else:
        ax1.plot(s_points, all_eigenvalues[:, i], 'gray', linewidth=0.5, alpha=0.3)

# Highlight minimum gap location
ax1.axvline(s_at_min, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Min gap at s={s_at_min:.3f}')

ax1.set_xlabel('s (interpolation parameter)', fontsize=14)
ax1.set_ylabel('Energy', fontsize=14)
ax1.set_title(f'Full Energy Spectrum Evolution (N={N}, {2**N} levels)', 
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='best')

# Bottom left: Zoom on lowest eigenvalues
ax2 = plt.subplot(2, 2, 3)
num_to_show = min(10, 2**N)
for i in range(num_to_show):
    if i == 0:
        ax2.plot(s_points, all_eigenvalues[:, i], 'b-', linewidth=2, label='E₀')
    elif i == 1:
        ax2.plot(s_points, all_eigenvalues[:, i], 'r-', linewidth=2, label='E₁')
    else:
        ax2.plot(s_points, all_eigenvalues[:, i], linewidth=1.5, alpha=0.7, 
                label=f'E_{i}')

ax2.axvline(s_at_min, color='green', linestyle='--', alpha=0.7)
ax2.set_xlabel('s', fontsize=12)
ax2.set_ylabel('Energy', fontsize=12)
ax2.set_title(f'Lowest {num_to_show} Energy Levels', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, ncol=2)

# Bottom right: Spectral gap
ax3 = plt.subplot(2, 2, 4)
ax3.plot(s_points, gaps, 'purple', linewidth=2.5)
ax3.axhline(min_gap, color='orange', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Δ_min = {min_gap:.6f}')
ax3.axvline(s_at_min, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax3.fill_between(s_points, 0, gaps, alpha=0.3, color='purple')
ax3.scatter([s_at_min], [min_gap], color='red', s=100, zorder=5, 
           label=f's = {s_at_min:.3f}')
ax3.set_xlabel('s', fontsize=12)
ax3.set_ylabel('Δ(s) = E₁ - E₀', fontsize=12)
ax3.set_title('Spectral Gap Evolution', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_ylim(bottom=-0.05*np.max(gaps))

plt.tight_layout()

# Save the plot
filename = f"full_spectrum_N{N}.png"
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f"\n💾 Plot saved to: {filename}")
plt.show()

# Additional analysis: Level statistics at different s values
print(f"\n{'='*70}")
print("Energy level statistics at key points:")
print(f"{'='*70}")

for s_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    idx = np.argmin(np.abs(s_points - s_val))
    evals = all_eigenvalues[idx, :]
    gap = evals[1] - evals[0]
    print(f"\ns = {s_val:.2f}:")
    print(f"  Energy range: [{evals.min():.4f}, {evals.max():.4f}]")
    print(f"  Spectral gap: {gap:.6f}")
    print(f"  Ground state: E₀ = {evals[0]:.4f}")
    print(f"  First excited: E₁ = {evals[1]:.4f}")
    print(f"  Lowest 5 levels: {evals[:5]}")

print(f"\n{'='*70}")
print("✅ Full spectrum analysis complete!")
print(f"{'='*70}")
