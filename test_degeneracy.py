#!/usr/bin/env python3
"""
Test script to demonstrate ground state degeneracy in Max-Cut 
and verify the fixed gap calculation.
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

N = 6
DEGENERACY_TOL = 1e-8

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

def find_first_gap(eigenvalues, tol=DEGENERACY_TOL):
    """Find gap to first non-degenerate excited state."""
    E0 = eigenvalues[0]
    degeneracy = 1
    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - E0) < tol:
            degeneracy += 1
        else:
            gap = eigenvalues[i] - E0
            return gap, degeneracy, eigenvalues[i]
    return 0.0, len(eigenvalues), E0

print("="*70)
print("  DEGENERACY TEST: Max-Cut Ground State Symmetry")
print("="*70)

# Generate a 3-regular graph
G = nx.random_regular_graph(d=3, n=N, seed=42)
edges = list(G.edges())
print(f"\nGraph: {N} nodes, {G.number_of_edges()} edges")
print(f"Edges: {edges}")

# Build Hamiltonians
H_B = build_H_initial(N)
H_P = build_H_problem(N, edges)

print(f"\n{'='*70}")
print("Analyzing eigenvalue spectrum at s=1 (pure problem Hamiltonian)")
print(f"{'='*70}")

# Get eigenvalues at s=1
all_evals = np.linalg.eigvalsh(H_P)

print(f"\nLowest 15 eigenvalues at s=1:")
for i in range(min(15, len(all_evals))):
    print(f"  E_{i:2d}: {all_evals[i]:10.6f}")

# Find degeneracies
print(f"\n{'='*70}")
print("Ground state degeneracy analysis:")
print(f"{'='*70}")

E0 = all_evals[0]
print(f"\nGround state energy: E₀ = {E0:.6f}")

# Count ground state degeneracy
gs_degeneracy = 0
for i, E in enumerate(all_evals):
    if abs(E - E0) < DEGENERACY_TOL:
        gs_degeneracy += 1
        print(f"  E_{i}: {E:.10f}  (degenerate with E₀)")
    else:
        print(f"  E_{i}: {E:.10f}  ← FIRST NON-DEGENERATE STATE")
        first_excited = E
        break

print(f"\n❗ Ground state degeneracy: {gs_degeneracy}")

# Compare old vs new gap calculation
naive_gap = all_evals[1] - all_evals[0]
correct_gap, deg, E_first = find_first_gap(all_evals)

print(f"\n{'='*70}")
print("Gap calculation comparison:")
print(f"{'='*70}")
print(f"\n❌ WRONG (naive): Δ = E₁ - E₀ = {naive_gap:.10f}")
print(f"   (Doesn't account for degeneracy!)")
print(f"\n✅ CORRECT: Δ = E_{{first non-deg}} - E₀ = {correct_gap:.10f}")
print(f"   Ground state has {deg} degenerate levels")
print(f"   First non-degenerate state: E_{deg} = {E_first:.6f}")

# Now scan through s and show where degeneracy matters
print(f"\n{'='*70}")
print("Scanning spectral gap from s=0 to s=1:")
print(f"{'='*70}")

s_points = np.linspace(0, 1, 50)
gaps_naive = []
gaps_correct = []
degeneracies = []

for s in s_points:
    H_s = (1-s) * H_B + s * H_P
    k_vals = min(10, H_s.shape[0])
    evals = eigh(H_s, eigvals_only=True, subset_by_index=(0, k_vals-1))
    
    # Naive: just E1 - E0
    gaps_naive.append(evals[1] - evals[0])
    
    # Correct: first non-degenerate gap
    gap, deg, _ = find_first_gap(evals)
    gaps_correct.append(gap)
    degeneracies.append(deg)

gaps_naive = np.array(gaps_naive)
gaps_correct = np.array(gaps_correct)
degeneracies = np.array(degeneracies)

# Find where they differ significantly
diff = np.abs(gaps_correct - gaps_naive)
significant_diff = diff > 1e-6

print(f"\nPoints where naive and correct gaps differ significantly:")
for i in np.where(significant_diff)[0]:
    print(f"  s={s_points[i]:.3f}: Naive={gaps_naive[i]:.6f}, "
          f"Correct={gaps_correct[i]:.6f}, Deg={degeneracies[i]}")

# Plot comparison
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Gap comparison
ax1.plot(s_points, gaps_naive, 'r--', linewidth=2, label='Naive: E₁ - E₀', alpha=0.7)
ax1.plot(s_points, gaps_correct, 'b-', linewidth=2, label='Correct: First non-deg gap')
ax1.fill_between(s_points, gaps_naive, gaps_correct, alpha=0.3, color='yellow',
                 label='Difference')
ax1.set_ylabel('Spectral Gap', fontsize=12)
ax1.set_title(f'Spectral Gap: Naive vs Correct Calculation (N={N})', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Difference plot
ax2.plot(s_points, diff, 'purple', linewidth=2)
ax2.fill_between(s_points, 0, diff, alpha=0.3, color='purple')
ax2.set_ylabel('|Correct - Naive|', fontsize=12)
ax2.set_title('Absolute Difference', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_yscale('log')

# Degeneracy
ax3.plot(s_points, degeneracies, 'g-', linewidth=2, marker='o', markersize=4)
ax3.fill_between(s_points, 1, degeneracies, alpha=0.3, color='green')
ax3.set_xlabel('s (interpolation parameter)', fontsize=12)
ax3.set_ylabel('Ground State Degeneracy', fontsize=12)
ax3.set_title('Ground State Degeneracy vs s', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1)
ax3.set_ylim(bottom=0.5)

plt.tight_layout()
filename = f"degeneracy_test_N{N}.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\n📊 Plot saved to: {filename}")
plt.show()

print(f"\n{'='*70}")
print("Summary:")
print(f"{'='*70}")
print(f"\n✅ At s=1, ground state has {gs_degeneracy}-fold degeneracy")
print(f"✅ Naive gap (E₁-E₀): {naive_gap:.10f}")
print(f"✅ Correct gap (to first non-deg): {correct_gap:.10f}")
print(f"✅ Difference: {abs(correct_gap - naive_gap):.10f}")
print(f"\n⚠️  Using naive gap would give WRONG results!")
print(f"✅ Fixed version correctly handles degeneracy")
print(f"\n{'='*70}")
