#!/usr/bin/env python3
"""
=========================================================================
Weighted Graph Spectrum Visualization
=========================================================================
Plot the full energy spectrum for a weighted graph from the weighted gap
analysis results. This helps investigate:
1. Whether minimum gaps near s=1 are real avoided crossings or degeneracy lifting
2. How random weights affect the ground state degeneracy at s=1
=========================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast
import re

from aqc_spectral_utils import (
    build_H_initial_sparse,
    build_H_problem_sparse_weighted,
    find_first_gap,
    load_graphs_from_file,
    DEGENERACY_TOL
)
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from output_config import get_run_dirs, save_file, save_run_info

# ============================================================================
# CONFIGURATION - Edit these parameters to visualize different trials
# ============================================================================
RESULTS_CSV = "outputs/qaoa_weighted/N12/weighted_gap_analysis_N12.csv"
GRAPH_FILE = "graphs_rawdata/12_3_3.scd"
GRAPH_ID = 10      # Which graph to visualize
TRIAL = 10           # Which trial (the one with gap 0.024432)
N = 12              # Number of qubits
S_RESOLUTION = 100  # Number of points for s interpolation
MAX_EIGENVALUES_TO_PLOT = 20  # Show only first N eigenvalues
# ============================================================================


def parse_weights_string(weights_str: str) -> list:
    """
    Parse weights string from CSV which contains np.float64() wrappers.
    Example: "[np.float64(0.534), np.float64(1.993), ...]"
    """
    # Extract all float values using regex
    pattern = r'np\.float64\(([\d.e+-]+)\)'
    matches = re.findall(pattern, weights_str)
    return [float(m) for m in matches]


def compute_spectrum_evolution_weighted(H_B, H_P, s_points, k_eigenvalues=None):
    """
    Compute eigenvalue evolution for weighted H(s) across interpolation points.

    Args:
        H_B: Sparse initial Hamiltonian
        H_P: Sparse weighted problem Hamiltonian
        s_points: Array of s values
        k_eigenvalues: If specified, compute only k lowest eigenvalues (sparse)

    Returns:
        Array of eigenvalues at each s point
    """
    if k_eigenvalues is not None:
        # Sparse method - only k lowest eigenvalues
        all_eigenvalues = np.zeros((len(s_points), k_eigenvalues))
        for i, s in enumerate(s_points):
            H_s = (1 - s) * H_B + s * H_P
            evals = eigsh(H_s, k=k_eigenvalues, which='SA', return_eigenvectors=False)
            all_eigenvalues[i, :] = np.sort(evals)
    else:
        # Dense method - all eigenvalues
        dim = H_B.shape[0]
        all_eigenvalues = np.zeros((len(s_points), dim))
        for i, s in enumerate(s_points):
            H_s = (1 - s) * H_B + s * H_P
            all_eigenvalues[i, :] = eigh(H_s.toarray(), eigvals_only=True)

    return all_eigenvalues


def main():
    print("=" * 70)
    print("  WEIGHTED GRAPH SPECTRUM VISUALIZATION")
    print("=" * 70)

    # Load trial data from CSV
    print(f"\n📖 Loading trial data from {RESULTS_CSV}")
    print(f"   Graph ID: {GRAPH_ID}, Trial: {TRIAL}")

    df = pd.read_csv(RESULTS_CSV)
    trial_row = df[(df['Graph_ID'] == GRAPH_ID) & (df['Trial'] == TRIAL)]

    if len(trial_row) == 0:
        raise ValueError(f"Trial not found: Graph_ID={GRAPH_ID}, Trial={TRIAL}")

    trial_row = trial_row.iloc[0]

    # Extract trial info
    original_gap = trial_row['Original_Delta_min']
    weighted_gap = trial_row['Weighted_Delta_min']
    s_at_min = trial_row['s_at_min']
    gap_change = trial_row['Gap_Change_Percent']
    weights_str = trial_row['Weights']

    # Parse weights
    weights = parse_weights_string(weights_str)

    print(f"\n📊 Trial Information:")
    print(f"   • Original (unweighted) Δ_min: {original_gap:.6f}")
    print(f"   • Weighted Δ_min: {weighted_gap:.6f}")
    print(f"   • s at minimum: {s_at_min:.4f}")
    print(f"   • Gap change: {gap_change:+.1f}%")
    print(f"   • Number of edge weights: {len(weights)}")
    print(f"   • Weight range: [{min(weights):.4f}, {max(weights):.4f}]")

    # Load graph edges
    print(f"\n📖 Loading graph edges from {GRAPH_FILE}")
    all_graphs = load_graphs_from_file(GRAPH_FILE)
    edges = all_graphs[GRAPH_ID - 1]  # Graph_ID is 1-indexed

    print(f"   • Graph has {len(edges)} edges")

    # Create networkx graph for visualization
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    # Build Hamiltonians
    print(f"\n🔨 Building Hamiltonians...")
    H_B = build_H_initial_sparse(N)
    H_P_weighted = build_H_problem_sparse_weighted(N, edges, weights)
    print("   ✓ Done")

    # Check degeneracy at s=1 (problem Hamiltonian)
    print(f"\n🔍 Checking degeneracy at s=1 (weighted problem Hamiltonian)...")
    diagonal = H_P_weighted.diagonal()
    k_check = min(10, len(diagonal))
    evals_s1 = np.partition(diagonal, k_check - 1)[:k_check]
    evals_s1 = np.sort(evals_s1)

    gap_s1, degeneracy_s1 = find_first_gap(evals_s1, tol=DEGENERACY_TOL)

    print(f"   • Ground state degeneracy at s=1: {degeneracy_s1}")
    print(f"   • First few eigenvalues at s=1: {evals_s1[:5]}")
    print(f"   • Gap E_{degeneracy_s1} - E_0 at s=1: {gap_s1:.6f}")

    if degeneracy_s1 == 1:
        print(f"   ⚠️  WARNING: Z2 symmetry is BROKEN - ground state is non-degenerate!")
        print(f"   ⚠️  The gap should be computed as E_1 - E_0, not E_2 - E_0")
    elif degeneracy_s1 == 2:
        print(f"   ✓ Z2 symmetry preserved - ground state has 2-fold degeneracy")
    else:
        print(f"   ⚠️  Unusual degeneracy: {degeneracy_s1}")

    # Compute spectrum evolution
    s_points = np.linspace(0, 1, S_RESOLUTION)

    print(f"\n🔬 Computing spectrum evolution...")
    print(f"   • {S_RESOLUTION} points in s ∈ [0, 1]")
    print(f"   • Computing {MAX_EIGENVALUES_TO_PLOT} lowest eigenvalues")

    all_eigenvalues = compute_spectrum_evolution_weighted(
        H_B, H_P_weighted, s_points, k_eigenvalues=MAX_EIGENVALUES_TO_PLOT
    )
    print("   ✓ Done")

    # Find minimum gap in computed spectrum
    # For proper gap calculation, we need to account for degeneracy
    if degeneracy_s1 == 1:
        # Non-degenerate: gap is E_1 - E_0
        gaps = all_eigenvalues[:, 1] - all_eigenvalues[:, 0]
        excited_level = 1
    else:
        # Degenerate: gap is E_k - E_0 where k = degeneracy
        k = min(degeneracy_s1, all_eigenvalues.shape[1] - 1)
        gaps = all_eigenvalues[:, k] - all_eigenvalues[:, 0]
        excited_level = k

    min_gap_idx = np.argmin(gaps)
    computed_min_gap = gaps[min_gap_idx]
    computed_s_at_min = s_points[min_gap_idx]

    print(f"\n📊 Spectrum Analysis:")
    print(f"   • Correct gap definition (based on degeneracy): E_{excited_level} - E_0")
    print(f"   • Computed minimum gap: {computed_min_gap:.6f}")
    print(f"   • Computed s at minimum: {computed_s_at_min:.4f}")
    print(f"   • Original analysis gap: {weighted_gap:.6f} at s={s_at_min:.4f}")

    if degeneracy_s1 == 1 and abs(computed_s_at_min - s_at_min) > 0.1:
        print(f"\n   ⚠️  DISCREPANCY: The original analysis used E_2 - E_0,")
        print(f"      but with broken symmetry, E_1 - E_0 is the correct gap!")

    # Create the plot
    print(f"\n📈 Creating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Full spectrum evolution
    ax1 = axes[0]

    # Plot all eigenvalues in gray first
    for i in range(all_eigenvalues.shape[1]):
        if i not in [0, excited_level]:
            ax1.plot(s_points, all_eigenvalues[:, i],
                    color='darkgray', linewidth=0.8, alpha=0.5,
                    label=f'E_{i}' if i < 4 else None)

    # Highlight ground state and minimum gap level
    ax1.plot(s_points, all_eigenvalues[:, 0], 'b-', linewidth=2.5, alpha=0.9,
            label=f'E_0 (ground state)')
    ax1.plot(s_points, all_eigenvalues[:, excited_level], 'r-', linewidth=2.5, alpha=0.9,
            label=f'E_{excited_level} (first excited)')

    # If degeneracy = 1, also show E_2 to illustrate the issue
    if degeneracy_s1 == 1 and all_eigenvalues.shape[1] > 2:
        ax1.plot(s_points, all_eigenvalues[:, 2], 'orange', linewidth=1.5, alpha=0.7,
                linestyle='--', label=f'E_2 (original analysis used this)')

    # Mark the minimum gap locations
    ax1.axvline(computed_s_at_min, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Correct min gap at s={computed_s_at_min:.3f}')

    if abs(s_at_min - computed_s_at_min) > 0.05:
        ax1.axvline(s_at_min, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Original analysis s={s_at_min:.3f}')

    # Add gap annotation
    gap_e0 = all_eigenvalues[min_gap_idx, 0]
    gap_e1 = all_eigenvalues[min_gap_idx, excited_level]
    ax1.annotate('', xy=(computed_s_at_min, gap_e1), xytext=(computed_s_at_min, gap_e0),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(computed_s_at_min + 0.02, (gap_e0 + gap_e1) / 2,
            f'Δ={computed_min_gap:.4f}', fontsize=10, color='green', fontweight='bold')

    ax1.set_xlabel('s (interpolation parameter)', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Weighted Spectrum - Graph {GRAPH_ID}, Trial {TRIAL}\n'
                  f'Degeneracy at s=1: {degeneracy_s1}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_xlim(0, 1)

    # Right plot: Gap evolution comparison
    ax2 = axes[1]

    # Plot E_1 - E_0 gap
    gap_1_0 = all_eigenvalues[:, 1] - all_eigenvalues[:, 0]
    ax2.plot(s_points, gap_1_0, 'b-', linewidth=2, label='E₁ - E₀ (correct if deg=1)')

    # Plot E_2 - E_0 gap (what original analysis used)
    if all_eigenvalues.shape[1] > 2:
        gap_2_0 = all_eigenvalues[:, 2] - all_eigenvalues[:, 0]
        ax2.plot(s_points, gap_2_0, 'r--', linewidth=2, label='E₂ - E₀ (original analysis)')

    # Mark where each has minimum
    min_idx_1 = np.argmin(gap_1_0)
    ax2.axvline(s_points[min_idx_1], color='blue', linestyle=':', alpha=0.5)
    ax2.scatter([s_points[min_idx_1]], [gap_1_0[min_idx_1]], color='blue', s=100, zorder=5)
    ax2.annotate(f'min={gap_1_0[min_idx_1]:.4f}\ns={s_points[min_idx_1]:.3f}',
                xy=(s_points[min_idx_1], gap_1_0[min_idx_1]),
                xytext=(s_points[min_idx_1] - 0.15, gap_1_0[min_idx_1] + 0.1),
                fontsize=9, color='blue')

    if all_eigenvalues.shape[1] > 2:
        min_idx_2 = np.argmin(gap_2_0)
        ax2.axvline(s_points[min_idx_2], color='red', linestyle=':', alpha=0.5)
        ax2.scatter([s_points[min_idx_2]], [gap_2_0[min_idx_2]], color='red', s=100, zorder=5)
        ax2.annotate(f'min={gap_2_0[min_idx_2]:.4f}\ns={s_points[min_idx_2]:.3f}',
                    xy=(s_points[min_idx_2], gap_2_0[min_idx_2]),
                    xytext=(s_points[min_idx_2] + 0.02, gap_2_0[min_idx_2] + 0.05),
                    fontsize=9, color='red')

    ax2.set_xlabel('s (interpolation parameter)', fontsize=12)
    ax2.set_ylabel('Energy Gap', fontsize=12)
    ax2.set_title('Gap Evolution Comparison\n'
                  f'Original analysis reported: Δ={weighted_gap:.4f} at s={s_at_min:.3f}',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, None)

    plt.tight_layout()

    # Save figure
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    output_filename = f"outputs/figures/weighted_spectrum_graph{GRAPH_ID}_trial{TRIAL}.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    print(f"\n💾 Plot saved to: {output_filename}")

    # Desktop mirror copy
    try:
        _, desktop_dir = get_run_dirs("weighted_spectrum", timestamp=True)
        save_file(output_filename, "weighted_spectrum", _desktop_dir=desktop_dir)
        save_run_info(desktop_dir, "weighted_spectrum", extra_info={"graph_id": GRAPH_ID, "trial": TRIAL})
    except Exception as e:
        print(f"  ⚠️ Desktop copy skipped: {e}")

    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Graph {GRAPH_ID}, Trial {TRIAL}:")
    print(f"  • Ground state degeneracy at s=1: {degeneracy_s1}")
    if degeneracy_s1 == 1:
        print(f"  • Z2 symmetry: BROKEN by weights")
        print(f"  • Correct gap: E₁ - E₀ = {gap_1_0[min_idx_1]:.6f} at s={s_points[min_idx_1]:.4f}")
        print(f"  • Original (wrong) gap: E₂ - E₀ = {weighted_gap:.6f} at s={s_at_min:.4f}")
    else:
        print(f"  • Z2 symmetry: PRESERVED")
        print(f"  • Gap: E₂ - E₀ = {computed_min_gap:.6f} at s={computed_s_at_min:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

