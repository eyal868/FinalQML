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
import pandas as pd
import time

from aqc_spectral_utils import (
    DEGENERACY_TOL,
    build_H_initial,
    build_H_problem,
    get_aqc_hamiltonian,
    find_first_gap,
    find_min_gap_with_degeneracy,
    find_min_gap_sparse,
    load_graphs_from_file,
    extract_graph_params
)

# =========================================================================
# 1. CONFIGURATION PARAMETERS
# =========================================================================

CONFIG = {
    'N_values': [14],                     # Which N to process: [10], [12], or [10, 12]
    'S_resolution': 100,                   # Sampling points along s ∈ [0, 1] (only used for 'grid' method)
    'graphs_per_N': {                     # Graph selection per N (1-indexed Graph_IDs from CSV)
        10: None,                         # None=all, int=first N, range/list=specific Graph_IDs
        12: None,                        # Use Graph_ID values from CSV (e.g.,  for deg=10)
        14: None,
        16: None
    },
    'k_vals_check': 50,                   # Max degeneracy threshold (skip graphs with deg ≥ this)
    'output_suffix': '-new_method_27_dec-scd',        # Optional filename suffix
    'degree': 3,                          # Regularity of graphs (e.g., 3 for 3-regular, 4 for 4-regular)
    
    # METHOD SELECTION: 'sparse' (recommended) or 'grid' (legacy)
    # 'sparse': Uses sparse eigensolvers + scalar optimization (fast, accurate, scales to N≥14)
    # 'grid': Uses dense diagonalization on fixed grid (slow, may miss true minimum)
    'method': 'sparse',
    
    # SPARSE METHOD OPTIONS (only used when method='sparse')
    'target_degeneracy': 2,               # Only process graphs with this degeneracy (k=2 for unique solutions)
    's_bounds': (0.01, 0.99),             # Optimization bounds (avoid exact 0 and 1)
    'xtol': 1e-4,                         # Tolerance for s location in optimization
}

# GENREG data files (supports both .asc and .scd formats)
GENREG_FILES = {
    10: {
        3: 'graphs_rawdata/10_3_3.asc',
        4: 'graphs_rawdata/10_4_3.asc',
        5: 'graphs_rawdata/10_5_3.asc'
    },
    12: {
        3: 'graphs_rawdata/12_3_3.scd'
    },
    14: {
        3: 'graphs_rawdata/14_3_3.scd'
    },
    16: {
        3: 'graphs_rawdata/16_3_3.scd'
    }
}

def _generate_output_filename():
    """Auto-generate filename from configuration."""
    N_str = f"N{CONFIG['N_values'][0]}" if len(CONFIG['N_values']) == 1 else f"N{'_'.join(map(str, CONFIG['N_values']))}"
    degree_str = f"{CONFIG['degree']}_regular"
    method = CONFIG.get('method', 'grid')
    if method == 'sparse':
        method_str = f"_sparse_k{CONFIG.get('target_degeneracy', 2)}"
    else:
        method_str = f"_res{CONFIG['S_resolution']}"
    return f"outputs/Delta_min_{degree_str}_{N_str}{method_str}{CONFIG['output_suffix']}.csv"

OUTPUT_FILENAME = _generate_output_filename()

# =========================================================================
# 2. MAIN EXECUTION
# =========================================================================

def main():
    """Main execution: process selected graphs from GENREG enumeration."""
    method = CONFIG.get('method', 'grid')
    use_sparse = (method == 'sparse')
    
    print("=" * 70)
    print("  AQC SPECTRAL GAP ANALYSIS FOR REGULAR GRAPHS")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  • N values: {CONFIG['N_values']}")
    print(f"  • Method: {method.upper()}" + (" (sparse eigensolver + Brent optimization)" if use_sparse else " (dense diagonalization on fixed grid)"))
    if use_sparse:
        print(f"  • Target degeneracy: k={CONFIG.get('target_degeneracy', 2)}")
        print(f"  • s bounds: {CONFIG.get('s_bounds', (0.01, 0.99))}")
        print(f"  • Optimization tolerance: {CONFIG.get('xtol', 1e-4)}")
    else:
        print(f"  • S_RESOLUTION: {CONFIG['S_resolution']}")
        print(f"  • k_vals_check: {CONFIG['k_vals_check']}")
    print(f"  • Degeneracy tolerance: {DEGENERACY_TOL}")
    print(f"  • Output: {OUTPUT_FILENAME}")
    print(f"  • Graph degree: {CONFIG['degree']}-regular")
    print(f"\n📁 Graph Selection:")
    for N in CONFIG['N_values']:
        selection = CONFIG['graphs_per_N'].get(N, None)
        if selection is None:
            # Update: reference GENREG_FILES with degree
            print(f"  • N={N}: All graphs from {GENREG_FILES.get(N, {}).get(CONFIG['degree'], 'N/A')}")
        elif isinstance(selection, int):
            print(f"  • N={N}: First {selection} graphs from {GENREG_FILES.get(N, {}).get(CONFIG['degree'], 'N/A')}")
        elif isinstance(selection, range):
            print(f"  • N={N}: Graph_IDs {selection.start}-{selection.stop-1} from {GENREG_FILES.get(N, {}).get(CONFIG['degree'], 'N/A')}")
        else:
            print(f"  • N={N}: Graph_IDs {sorted(selection)} from {GENREG_FILES.get(N, {}).get(CONFIG['degree'], 'N/A')}")
    
    # Initialize
    s_points = np.linspace(0.0, 1.0, CONFIG['S_resolution']) if not use_sparse else None
    data = []
    total_eigsh_calls = 0  # Track sparse method efficiency
    
    # Start timer
    start_time = time.time()
    print(f"\n🚀 Starting analysis...")
    print("-" * 70)
    
    graph_counter = 0 # Initialize graph_counter
    skipped_degeneracy = 0  # Track graphs skipped due to degeneracy mismatch

    total_graphs = sum(
        len([i for i in (CONFIG['graphs_per_N'].get(N, None) or range(len(load_graphs_from_file(GENREG_FILES[N][CONFIG['degree']]))))
             if isinstance(CONFIG['graphs_per_N'].get(N, None), (range, list, set)) or i < (CONFIG['graphs_per_N'].get(N, None) or len(load_graphs_from_file(GENREG_FILES[N][CONFIG['degree']])))])
        if N in GENREG_FILES else 0 for N in CONFIG['N_values']
    ) if all(N in GENREG_FILES for N in CONFIG['N_values']) else 0
    
    # Process configured N values
    for N in sorted(CONFIG['N_values']):
        degree = CONFIG['degree'] # Get degree from config
        
        if N not in GENREG_FILES or degree not in GENREG_FILES[N]:
            print(f"\n⚠️  Skipping N={N}, deg={degree}: No GENREG file specified")
            continue
            
        filename = GENREG_FILES[N][degree] # Retrieve filename using N and degree
        print(f"\n▶ Processing N={N}, deg={degree} (Hilbert space dimension: 2^{N} = {2**N})")
        print(f"  📖 Reading graphs from {filename}...", end=" ")
        
        # Parse all graphs from file (supports .asc and .scd)
        try:
            all_graphs = load_graphs_from_file(filename)
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
        
        num_edges = degree * N // 2  # Calculate num_edges dynamically based on degree
        
        if use_sparse:
            # SPARSE METHOD: No need to pre-build dense H_B
            print(f"  🔧 Using SPARSE method (Lanczos + Brent optimization)")
        else:
            # GRID METHOD: Pre-calculate the dense Initial Hamiltonian (H_B) for this N
            print(f"  🔨 Building dense H_initial matrix {2**N}×{2**N}...", end=" ")
            H_B = build_H_initial(N)
            print("✓")
        
        # Process each graph sequentially
        graph_start_time = time.time()
        for i, (file_pos, edges) in enumerate(zip(graph_ids, graphs)):
            graph_counter += 1
            iter_start = time.time()
            
            # Graph_ID is 1-indexed file position (not sequential counter)
            graph_id = file_pos + 1
            
            if use_sparse:
                # SPARSE METHOD: Use optimized sparse eigensolver + scalar optimization
                result = find_min_gap_sparse(
                    N=N,
                    edges=edges,
                    num_edges=num_edges,
                    target_degeneracy=CONFIG.get('target_degeneracy', 2),
                    s_bounds=CONFIG.get('s_bounds', (0.01, 0.99)),
                    xtol=CONFIG.get('xtol', 1e-4),
                    verbose=False
                )
                delta_min, s_min, max_deg, max_cut, num_evals = result
                total_eigsh_calls += num_evals
            else:
                # GRID METHOD: Use dense diagonalization on fixed grid
                H_P = build_H_problem(N, edges)
                result = find_min_gap_with_degeneracy(H_B, H_P, s_points, num_edges, 
                                                       CONFIG['k_vals_check'], verbose=True)
                delta_min, s_min, max_deg, max_cut, _ = result
            
            # Skip graph if degeneracy doesn't match (sparse) or exceeds threshold (grid)
            if delta_min is None:
                skipped_degeneracy += 1
                if use_sparse:
                    print(f"    [{i+1:3d}/{len(graphs)}] Graph_ID={graph_id:3d} | SKIPPED (deg ≠ {CONFIG.get('target_degeneracy', 2)})")
                else:
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
            if use_sparse:
                print(f"    [{i+1:3d}/{len(graphs)}] Graph_ID={graph_id:3d} | "
                      f"⏱️  {graph_time:.2f}s | Δ_min={delta_min:.6f} s={s_min:.4f} "
                      f"cut={max_cut} ({num_evals} eigsh)")
            else:
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
    print(f"   • Method: {method.upper()}")
    print(f"   • Total graphs processed: {len(data)}")
    print(f"   • Graphs skipped (degeneracy): {skipped_degeneracy}")
    print(f"   • Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if len(data) > 0:
        print(f"   • Average time per graph: {total_time/len(data):.2f} seconds")
        if use_sparse:
            print(f"   • Total eigsh calls: {total_eigsh_calls}")
            print(f"   • Average eigsh calls per graph: {total_eigsh_calls/len(data):.1f}")
    
    if len(data) > 0:
        print(f"\n📈 Spectral Gap Statistics (Overall):")
        print(f"   • Mean Δ_min: {df['Delta_min'].mean():.6f}")
        print(f"   • Std Δ_min: {df['Delta_min'].std():.6f}")
        print(f"   • Min Δ_min: {df['Delta_min'].min():.6f} (hardest instance)")
        print(f"   • Max Δ_min: {df['Delta_min'].max():.6f} (easiest instance)")
        
        print(f"\n📈 Statistics by N:")
        for N_val in sorted(CONFIG['N_values']):
            df_n = df[df['N'] == N_val]
            if len(df_n) > 0:
                print(f"   N={N_val:2d}: Count={len(df_n):3d}, "
                      f"Δ_min={df_n['Delta_min'].mean():.6f}±{df_n['Delta_min'].std():.6f}, "
                      f"MaxCut={df_n['Max_cut_value'].mean():.1f}±{df_n['Max_cut_value'].std():.1f}")
    else:
        print(f"\n⚠️  No graphs matched the degeneracy filter (k={CONFIG.get('target_degeneracy', 2)})")
    
    print(f"\n📁 Data saved to: {OUTPUT_FILENAME}")
    print("=" * 70)


if __name__ == "__main__":
    main()
