#!/usr/bin/env python3
"""
Plot the full energy spectrum for AQC MaxCut solution from data tables.
Automatically detects and highlights the eigenvalue levels with minimum gap.
Uses shared methodology from aqc_spectral_utils.py.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast

# Import shared AQC utilities
from aqc_spectral_utils import (
    build_H_initial,
    build_H_problem,
    analyze_spectrum_for_visualization,
    DEGENERACY_TOL
)

# ============================================================================
# CONFIGURATION - Edit these parameters to visualize different graphs
# ============================================================================
CSV_FILENAME = "DataOutputs/Delta_min_3_regular_N12_res20.csv"
GRAPH_ID = 44  # Which graph from the CSV to visualize
S_RESOLUTION = 50  # Number of points for s interpolation
MAX_EIGENVALUES_TO_PLOT = 50  # Show only first N eigenvalues (None for all)
# ============================================================================

def load_graph_from_csv(csv_filename, graph_id):
    """
    Load graph data from CSV file by graph ID.
    
    Returns:
        dict with keys: 'N', 'edges', 'Delta_min', 's_at_min', 'Max_degeneracy', 'Max_cut_value'
    """
    df = pd.read_csv(csv_filename)
    row = df[df['Graph_ID'] == graph_id]
    
    if len(row) == 0:
        raise ValueError(f"Graph ID {graph_id} not found in {csv_filename}")
    
    row = row.iloc[0]
    
    # Parse edges string (stored as Python list literal)
    edges = ast.literal_eval(row['Edges'])
    
    return {
        'N': int(row['N']),
        'edges': edges,
        'Delta_min': float(row['Delta_min']),
        's_at_min': float(row['s_at_min']),
        'Max_degeneracy': int(row['Max_degeneracy']),
        'Max_cut_value': int(row['Max_cut_value'])
    }

print("="*70)
print("  FULL SPECTRUM EVOLUTION - AQC MaxCut Solution")
print("="*70)

# Load graph from CSV
print(f"\nLoading graph from: {CSV_FILENAME}")
print(f"Graph ID: {GRAPH_ID}")
graph_data = load_graph_from_csv(CSV_FILENAME, GRAPH_ID)

N = graph_data['N']
edges = graph_data['edges']

# Create networkx graph for information
G = nx.Graph()
G.add_nodes_from(range(N))
G.add_edges_from(edges)

print(f"\nGraph: N={N} nodes, {G.number_of_edges()} edges")
print(f"Max cut value: {graph_data['Max_cut_value']}")
print(f"Max degeneracy: {graph_data['Max_degeneracy']}")
print(f"Expected Δ_min from CSV: {graph_data['Delta_min']:.6f} at s = {graph_data['s_at_min']:.3f}")

# Build Hamiltonians
print(f"\n🔨 Building Hamiltonians for N={N} system...")
H_B = build_H_initial(N)
H_P = build_H_problem(N, edges)
print("  ✓ Done")

# Compute full spectrum evolution with proper degeneracy handling
print(f"\n🔬 Analyzing spectrum evolution (following spectral_gap_analysis.py methodology)...")
print(f"   • Computing eigenvalues at {S_RESOLUTION} points along s ∈ [0,1]")
print(f"   • System dimension: {2**N}×{2**N}")

s_points = np.linspace(0, 1, S_RESOLUTION)
num_edges = len(edges)

# Use shared analysis function
analysis_result = analyze_spectrum_for_visualization(H_B, H_P, s_points, num_edges)

all_eigenvalues = analysis_result['all_eigenvalues']
min_gap = analysis_result['min_gap']
s_at_min = analysis_result['s_at_min']
min_gap_idx = analysis_result['min_gap_idx']
degeneracy = analysis_result['degeneracy']
min_gap_level1 = analysis_result['level1']
min_gap_level2 = analysis_result['level2']

print(f"  ✓ Analysis complete")

print(f"\n📊 Spectral Gap Analysis Results:")
print(f"  • Ground state degeneracy at s=1: {degeneracy}")
if degeneracy > 1:
    print(f"    → Ignoring E₁ through E_{degeneracy-1} (degenerate with E₀ at s=1)")
print(f"  • Minimum spectral gap: Δ_min = {min_gap:.6f}")
print(f"  • Gap location: s = {s_at_min:.3f}")
print(f"  • Gap between: E_{min_gap_level1} (ground) and E_{min_gap_level2}")
print(f"  • Comparison with CSV: {graph_data['Delta_min']:.6f} (expected)")
print(f"  • Match: {'✓' if abs(min_gap - graph_data['Delta_min']) < 0.001 else '✗ WARNING'}")

# Create the plot
print(f"\n📈 Creating visualization...")

# Determine which eigenvalues to plot
if MAX_EIGENVALUES_TO_PLOT is not None:
    eigenvalues_to_show = min(MAX_EIGENVALUES_TO_PLOT, 2**N)
    print(f"   • Plotting first {eigenvalues_to_show} eigenvalues (out of {2**N} total)")
    # Always ensure highlighted levels are visible
    if min_gap_level2 >= eigenvalues_to_show:
        eigenvalues_to_show = min_gap_level2 + 1
        print(f"   • Extended to {eigenvalues_to_show} to include min gap level E_{min_gap_level2}")
else:
    eigenvalues_to_show = 2**N
    print(f"   • Plotting all {2**N} eigenvalues")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot eigenvalues in gray (excluding highlighted ones)
for i in range(eigenvalues_to_show):
    if i not in [min_gap_level1, min_gap_level2]:
        ax.plot(s_points, all_eigenvalues[:, i], color='darkgray', linewidth=0.8, alpha=0.5)

# Highlight the detected minimum gap levels
ax.plot(s_points, all_eigenvalues[:, min_gap_level1], 'b-', linewidth=3, alpha=0.9, 
        label=f'E_{min_gap_level1} (ground state)')
ax.plot(s_points, all_eigenvalues[:, min_gap_level2], 'r-', linewidth=3, alpha=0.9, 
        label=f'E_{min_gap_level2} (min gap level)')

# Mark minimum gap location
ax.axvline(s_at_min, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Min gap at s={s_at_min:.3f}')

# Add arrow annotation showing the gap at minimum
gap_energy_1 = all_eigenvalues[min_gap_idx, min_gap_level1]
gap_energy_2 = all_eigenvalues[min_gap_idx, min_gap_level2]
ax.annotate('', xy=(s_at_min, gap_energy_2), xytext=(s_at_min, gap_energy_1),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(s_at_min + 0.02, (gap_energy_1 + gap_energy_2) / 2, 
        f'Δ={min_gap:.4f}', fontsize=11, color='green', fontweight='bold')

ax.set_xlabel('s (interpolation parameter)', fontsize=14)
ax.set_ylabel('Energy', fontsize=14)

# Update title based on whether we're showing all eigenvalues
title_str = f'Energy Spectrum Evolution - AQC MaxCut\n'
title_str += f'Graph ID={GRAPH_ID}, N={N}'
if eigenvalues_to_show < 2**N:
    title_str += f' (showing first {eigenvalues_to_show}/{2**N} eigenvalues)'
else:
    title_str += f' ({2**N} eigenvalues)'
    
ax.set_title(title_str, fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')
ax.set_xlim(0, 1)

plt.tight_layout()
filename = f"outputs/example_full_spectrum_N{N}_graph{GRAPH_ID}--------.png"
plt.savefig(filename, dpi=200, bbox_inches='tight')
print(f"\n📊 Plot saved to: {filename}")
plt.show()

print("\n" + "="*70)
print("✅ Visualization complete!")
print("="*70)
