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

CONFIG = {
    'N_values': [12],                     # Which N to process: [10], [12], or [10, 12]
    'S_resolution': 20,                   # Sampling points along s ∈ [0, 1]
    'graphs_per_N': {                     # Graph selection per N (1-indexed Graph_IDs from CSV)
        10: None,                         # None=all, int=first N, range/list=specific Graph_IDs
        12: None,  # Use Graph_ID values from CSV (e.g.,  for deg=10)
    },
    'k_vals_check': 5,                   # Eigenvalues to check for degeneracy
    'output_suffix': 'fixed_method'                # Optional filename suffix
}

# GENREG data files
GENREG_FILES = {10: '10_3_3.asc', 12: '12_3_3.asc'}
DEGENERACY_TOL = 1e-8
def _generate_output_filename():
    """Auto-generate filename from configuration."""
    N_str = f"N{CONFIG['N_values'][0]}" if len(CONFIG['N_values']) == 1 else f"N{'_'.join(map(str, CONFIG['N_values']))}"
    return f"outputs/Delta_min_3_regular_{N_str}_res{CONFIG['S_resolution']}{CONFIG['output_suffix']}.csv"

OUTPUT_FILENAME = _generate_output_filename()

# =========================================================================
# 2. PAULI MATRICES
# =========================================================================

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

# =========================================================================
# 3. HAMILTONIAN CONSTRUCTION
# =========================================================================

def pauli_tensor_product(op_list: List[np.ndarray]) -> np.ndarray:
    """Tensor product of 2x2 matrices."""
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result

def get_pauli_term(N: int, pauli_type: str, index1: int, index2: int = -1) -> np.ndarray:
    """Generate N-qubit operator: X_i or Z_i⊗Z_j."""
    operators = [IDENTITY] * N
    if pauli_type == 'X':
        operators[index1] = SIGMA_X
    elif pauli_type == 'ZZ':
        operators[index1] = SIGMA_Z
        operators[index2] = SIGMA_Z
    return pauli_tensor_product(operators)

def build_H_initial(N: int) -> np.ndarray:
    """H_initial = -∑ᵢ X̂ᵢ (transverse field mixer)"""
    H_B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        H_B += get_pauli_term(N, 'X', i)
    return -H_B

def build_H_problem(N: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ (Max-Cut Hamiltonian)"""
    H_P = np.zeros((2**N, 2**N), dtype=complex)
    for u, v in edges:
        H_P += get_pauli_term(N, 'ZZ', u, v)
    return H_P

def get_aqc_hamiltonian(s: float, H_B: np.ndarray, H_P: np.ndarray) -> np.ndarray:
    """H(s) = (1-s)·H_initial + s·H_problem"""
    return (1 - s) * H_B + s * H_P

# =========================================================================
# 4. GENREG FILE PARSING
# =========================================================================

def parse_asc_file(filename: str) -> List[List[Tuple[int, int]]]:
    """Parse GENREG .asc file, return list of graphs as 0-indexed edge lists."""
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
                in_adjacency = False
            elif in_adjacency and ':' in line:
                try:
                    parts = line.split(':')
                    vertex = int(parts[0].strip())
                    neighbors = [int(x) for x in parts[1].split()]
                    current_graph[vertex] = neighbors
                except (ValueError, IndexError):
                    continue
    
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
            edge = (min(v, n), max(v, n))
            if edge not in seen:
                seen.add(edge)
                edges.append((edge[0] - 1, edge[1] - 1))
    return sorted(edges)

# =========================================================================
# 5. SPECTRAL GAP CALCULATION
# =========================================================================

def find_first_gap(eigenvalues: np.ndarray, tol: float = DEGENERACY_TOL) -> Tuple[float, int]:
    """Find the gap between ground state and first non-degenerate excited state.
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
    Calculate minimum spectral gap with degeneracy-aware tracking.
    
    NEW METHODOLOGY (Fixed):
    1. Determines ground state degeneracy k at s=1
    2. For each s, computes ALL eigenvalues and finds minimum gap among
       all non-degenerate excited states: min(E[k:] - E[0])
    3. Returns the global minimum gap across all s values
    
    This ensures we don't miss the true minimum gap when different 
    excited states approach the ground state at different s values.
    """
    # Determine degeneracy at s=1 (problem Hamiltonian)
    H_final = H_P
    k_vals = min(CONFIG['k_vals_check'], H_final.shape[0])
    evals_final = eigh(H_final, eigvals_only=True, subset_by_index=(0, k_vals-1))
    _, degeneracy_s1 = find_first_gap(evals_final)
    
    # Filter: skip graphs where degeneracy exceeds k_vals_check threshold
    if degeneracy_s1 >= CONFIG['k_vals_check']:
        return None, None, None, None
    
    # Calculate max cut value from ground state energy
    # For H_problem = ∑_{(i,j)∈E} Z_i Z_j:
    # - Edges in cut contribute -1, edges not in cut contribute +1
    # - Cut_value = (total_edges - E_0) / 2
    E_0 = evals_final[0]
    max_cut_value = int((num_edges - E_0) / 2)
    
    # Track minimum gap across ALL excited states (not degenerate with ground)
    k_index = degeneracy_s1
    min_gap = np.inf
    s_at_min = 0.0

    for s in s_points:
        H_s = get_aqc_hamiltonian(s, H_B, H_P)
        
        # Compute ALL eigenvalues for maximum accuracy
        eigenvalues = eigh(H_s, eigvals_only=True)
        
        percent_complete = int((s / s_points[-1]) * 100)
        print(f"\r   Current graph progress: {percent_complete}% complete", end="")

        # Find minimum gap among all non-degenerate excited states
        # This checks ALL eigenvalues starting from k_index onwards
        gaps_at_s = eigenvalues[k_index:] - eigenvalues[0]
        gap_at_s = np.min(gaps_at_s)
        
        if gap_at_s < min_gap:
            min_gap = gap_at_s
            s_at_min = s
    
    return float(min_gap), float(s_at_min), int(degeneracy_s1), max_cut_value

# =========================================================================
# 6. MAIN EXECUTION
# =========================================================================

def main():
    """Main execution: process selected graphs from GENREG enumeration."""
    print("=" * 70)
    print("  AQC SPECTRAL GAP ANALYSIS FOR 3-REGULAR GRAPHS")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  • N values: {CONFIG['N_values']}")
    print(f"  • S_RESOLUTION: {CONFIG['S_resolution']}")
    print(f"  • k_vals_check: {CONFIG['k_vals_check']}")
    print(f"  • Degeneracy tolerance: {DEGENERACY_TOL}")
    print(f"  • Output: {OUTPUT_FILENAME}")
    print(f"\n📁 Graph Selection:")
    for N in CONFIG['N_values']:
        selection = CONFIG['graphs_per_N'].get(N, None)
        if selection is None:
            print(f"  • N={N}: All graphs from {GENREG_FILES.get(N, 'N/A')}")
        elif isinstance(selection, int):
            print(f"  • N={N}: First {selection} graphs from {GENREG_FILES.get(N, 'N/A')}")
        elif isinstance(selection, range):
            print(f"  • N={N}: Graph_IDs {selection.start}-{selection.stop-1} from {GENREG_FILES.get(N, 'N/A')}")
        else:
            print(f"  • N={N}: Graph_IDs {sorted(selection)} from {GENREG_FILES.get(N, 'N/A')}")
    
    # Initialize
    s_points = np.linspace(0.0, 1.0, CONFIG['S_resolution'])
    data = []
    
    # Start timer
    start_time = time.time()
    print(f"\n🚀 Starting analysis...")
    print("-" * 70)
    
    graph_counter = 0
    # Handling special specific graphs configs
    total_graphs = sum(
        len([i for i in (CONFIG['graphs_per_N'].get(N, None) or range(len(parse_asc_file(GENREG_FILES[N]))))
             if isinstance(CONFIG['graphs_per_N'].get(N, None), (range, list, set)) or i < (CONFIG['graphs_per_N'].get(N, None) or len(parse_asc_file(GENREG_FILES[N])))])
        if N in GENREG_FILES else 0 for N in CONFIG['N_values']
    ) if all(N in GENREG_FILES for N in CONFIG['N_values']) else 0
    
    # Process configured N values
    for N in sorted(CONFIG['N_values']):
        if N not in GENREG_FILES:
            print(f"\n⚠️  Skipping N={N}: No GENREG file specified")
            continue
            
        filename = GENREG_FILES[N]
        print(f"\n▶ Processing N={N} (Hilbert space dimension: 2^{N} = {2**N})")
        print(f"  📖 Reading graphs from {filename}...", end=" ")
        
        # Parse all graphs from file
        try:
            all_graphs = parse_asc_file(filename)
            print(f"✓ Found {len(all_graphs)} graphs in file")
        except FileNotFoundError:
            print(f"\n  ❌ Error: File not found: {filename}")
            continue
        except Exception as e:
            print(f"\n  ❌ Error parsing file: {e}")
            continue
        
        # Select which graphs to process based on configuration
        # Config values are Graph_IDs (1-indexed), convert to file positions (0-indexed)
        selection = CONFIG['graphs_per_N'].get(N, None)
        if selection is None:
            graphs = all_graphs
            graph_ids = list(range(len(all_graphs)))  # 0-indexed positions
        elif isinstance(selection, int):
            graphs = all_graphs[:selection]
            graph_ids = list(range(selection))
        elif isinstance(selection, range):
            # Config has Graph_IDs (1-indexed), convert to positions (0-indexed)
            positions = [i-1 for i in selection if 1 <= i <= len(all_graphs)]
            graphs = [all_graphs[i] for i in positions]
            graph_ids = positions
        else:  # set, list, or other iterable (contains Graph_IDs, 1-indexed)
            positions = [i-1 for i in sorted(selection) if 1 <= i <= len(all_graphs)]
            graphs = [all_graphs[i] for i in positions]
            graph_ids = positions
        
        print(f"  🎯 Processing {len(graphs)} selected graphs")
        
        # Pre-calculate the Initial Hamiltonian (H_B) for this N
        print(f"  🔨 Building H_initial matrix {2**N}×{2**N}...", end=" ")
        H_B = build_H_initial(N)
        print("✓")
        
        num_edges = 3 * N // 2  # For 3-regular graphs
        
        # Process each graph sequentially
        graph_start_time = time.time()
        for i, (file_pos, edges) in enumerate(zip(graph_ids, graphs)):
            graph_counter += 1
            iter_start = time.time()
            
            # Graph_ID is 1-indexed file position (not sequential counter)
            graph_id = file_pos + 1
            
            # Build the Problem Hamiltonian (H_P) for this specific graph
            H_P = build_H_problem(N, edges)
            
            # Calculate the Minimum Spectral Gap (handling degeneracy) and max cut
            delta_min, s_min, max_deg, max_cut = calculate_min_gap_robust(H_B, H_P, s_points, num_edges)
            
            # Skip graph if degeneracy exceeds threshold
            if delta_min is None:
                print(f"\n    [{i+1:3d}/{len(graphs)}] Graph_ID={graph_id:3d} | SKIPPED (deg ≥ {CONFIG['k_vals_check']})")
                continue
            
            # Store results
            data.append({
                'N': N,
                'Graph_ID': graph_id,  # 1-indexed file position
                'Delta_min': delta_min,
                's_at_min': s_min,
                'Max_degeneracy': max_deg,
                'Max_cut_value': max_cut,
                'Edges': str(edges)
            })
            
            # Timing for this graph
            graph_time = time.time() - iter_start
            elapsed_total = time.time() - start_time
            avg_time = elapsed_total / graph_counter
            eta = avg_time * (total_graphs - graph_counter)
            
            # Progress reporting (every graph)
            print(f"    [{i+1:3d}/{len(graphs)}] Graph_ID={graph_id:3d} | "
                  f"⏱️  {graph_time:.1f}s | Δ_min={delta_min:.6f} s={s_min:.2f} "
                  f"cut={max_cut} deg={max_deg} | Avg: {avg_time:.1f}s/graph | ETA: {eta/60:.1f}min")
    
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
        df_n = df[df['N'] == 12 ]
        if len(df_n) > 0:
            print(f"   N={N:2d}: Count={len(df_n):3d}, "
                  f"Δ_min={df_n['Delta_min'].mean():.6f}±{df_n['Delta_min'].std():.6f}, "
                  f"MaxCut={df_n['Max_cut_value'].mean():.1f}±{df_n['Max_cut_value'].std():.1f}")
    
    print(f"\n📁 Data saved to: {OUTPUT_FILENAME}")
    print("=" * 70)


if __name__ == "__main__":
    main()
