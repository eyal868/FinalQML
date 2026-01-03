#!/usr/bin/env python3
"""
=========================================================================
Spectral Gap Analysis for Regular Graphs (Optimized)
=========================================================================
Research Goal: Analyze the minimum energy gap (Δ_min) of the Adiabatic 
Quantum Computing (AQC) Hamiltonian H(s) for Max-Cut on regular graphs.

METHODOLOGY:
1. 'sparse': Uses sparse eigensolvers (Lanczos) with Brent's scalar optimization.
   - Ideal for N=12 to N=20.
   - Fast and accurate.
2. 'grid': Legacy method using dense diagonalization on a fixed grid.
   - Slow, restricted to N<=12.
=========================================================================
"""

import numpy as np
import pandas as pd
import time
from typing import Optional
from multiprocessing import Process, Queue

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
    'N_values': [14],                     # Which N to process: [10], [12], [14], etc.
    'S_resolution': 100,                  # ONLY for 'grid' method
    'graphs_per_N': {                     # Graph selection per N
        10: None,
        12: None,
        14: [236],#[242,475,410,251,373,82,434,247],
        16: [3218] #[125, 346, 418, 480, 487, 547, 911, 932, 1082, 1087, 1095, 1113, 1122, 1123, 1169, 1173, 1183, 1241, 1275, 1303, 1342, 1368, 1440, 1504, 1587, 1807, 1831, 1856, 1947, 1966, 1981, 1985, 1993, 2081, 2089, 2202, 2250, 2269, 2274, 2287, 2290, 2293, 2314, 2339, 2436, 2469, 2487, 2489, 2500, 2534, 2566, 2572, 2593, 2607, 2611, 2613, 2694, 2712, 2725, 2828, 2853, 2905, 2955, 3015, 3085, 3087, 3107, 3136, 3162, 3218, 3221, 3332, 3368, 3410, 3472, 3686, 3689, 3702, 3711, 3721, 3722, 3727, 3742, 3743, 3747, 3751, 3756, 3787, 3790, 3813, 3821, 3825, 3845, 3855, 3859, 3904, 3908, 3948, 3985, 4000, 4016, 4019]
    },
    'k_vals_check': 50,                   # Threshold for 'grid' method
    
    'output_suffix': '-sparse_optimized-new-skipped-check-best8', # Filename suffix
    'degree': 3,                          # Graph regularity (3, 4, 5)
    
    # METHOD SELECTION: 'sparse' (recommended) or 'grid' (legacy)
    'method': 'sparse',
    
    # SPARSE METHOD OPTIONS
    'target_degeneracy': 2,               # Only process k=2 graphs (unique solution)
    's_bounds': (0.01, 0.99),             # Optimization bounds
    'xtol': 1e-4,                         # Optimization tolerance
    
    # TIMEOUT
    'graph_timeout': 100,                   # Timeout in seconds per graph (None to disable)
}

# GENREG data files
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
# 2. TIMEOUT HELPER
# =========================================================================

def _worker_wrapper(queue, func, args, kwargs):
    """Worker function that runs in subprocess and puts result in queue."""
    try:
        result = func(*args, **kwargs)
        queue.put(('success', result))
    except Exception as e:
        queue.put(('error', str(e)))


def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    """
    Run a function with a timeout using multiprocessing.
    
    Args:
        func: Function to run
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        timeout: Timeout in seconds (None = no timeout)
        
    Returns:
        Tuple of (result, timed_out):
        - result: Function return value (or None if timed out/error)
        - timed_out: True if function was terminated due to timeout
    """
    if timeout is None:
        # No timeout, run directly
        return func(*args, **(kwargs or {})), False
    
    kwargs = kwargs or {}
    queue = Queue()
    
    p = Process(target=_worker_wrapper, args=(queue, func, args, kwargs))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        # Timeout occurred
        p.terminate()
        p.join()
        return None, True
    
    # Get result from queue
    if not queue.empty():
        status, result = queue.get()
        if status == 'success':
            return result, False
        else:
            # Error occurred in subprocess
            print(f"    Warning: Subprocess error: {result}")
            return None, False
    
    return None, False


# =========================================================================
# 3. MAIN EXECUTION
# =========================================================================

def main():
    """Main execution: process selected graphs."""
    method = CONFIG.get('method', 'grid')
    use_sparse = (method == 'sparse')
    
    print("=" * 70)
    print("  AQC SPECTRAL GAP ANALYSIS")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  • N values: {CONFIG['N_values']}")
    print(f"  • Method: {method.upper()}" + (" (Optimized Sparse + Brent's Optimization)" if use_sparse else " (Dense Grid)"))
    
    if use_sparse:
        print(f"  • Target degeneracy: k={CONFIG.get('target_degeneracy', 2)}")
        print(f"  • s bounds: {CONFIG.get('s_bounds', (0.01, 0.99))}")
        print(f"  • Optimization tolerance: {CONFIG.get('xtol', 1e-4)}")
        timeout = CONFIG.get('graph_timeout', None)
        print(f"  • Timeout per graph: {timeout}s" if timeout else "  • Timeout: disabled")
    else:
        print(f"  • S_RESOLUTION: {CONFIG['S_resolution']}")
        print(f"  • k_vals_check: {CONFIG['k_vals_check']}")
        
    print(f"  • Output: {OUTPUT_FILENAME}")
    print(f"  • Graph degree: {CONFIG['degree']}-regular")
    
    # Initialize
    s_points = np.linspace(0.0, 1.0, CONFIG['S_resolution']) if not use_sparse else None
    data = []
    total_eigsh_calls = 0
    skipped_degeneracy = 0
    timeout_skipped = []  # Track graphs skipped due to timeout
    graph_timeout = CONFIG.get('graph_timeout', None)
    
    start_time = time.time()
    print(f"\n🚀 Starting analysis...")
    print("-" * 70)
    
    graph_counter = 0
    
    # Process configured N values
    for N in sorted(CONFIG['N_values']):
        degree = CONFIG['degree']
        
        if N not in GENREG_FILES or degree not in GENREG_FILES[N]:
            print(f"\n⚠️  Skipping N={N}: No GENREG file specified")
            continue
            
        filename = GENREG_FILES[N][degree]
        print(f"\n▶ Processing N={N}, deg={degree} (Hilbert dim: {2**N})")
        print(f"  📖 Reading graphs from {filename}...", end=" ")
        
        try:
            all_graphs = load_graphs_from_file(filename)
            print(f"✓ Found {len(all_graphs)} graphs")
        except FileNotFoundError:
            print(f"\n  ❌ Error: File not found: {filename}")
            continue
        except Exception as e:
            print(f"\n  ❌ Error parsing file: {e}")
            continue
        
        # Select graphs
        selection = CONFIG['graphs_per_N'].get(N, None)
        if selection is None:
            graphs = all_graphs
            graph_ids = list(range(len(all_graphs)))
        elif isinstance(selection, int):
            graphs = all_graphs[:selection]
            graph_ids = list(range(selection))
        elif isinstance(selection, range):
            positions = [i-1 for i in selection if 1 <= i <= len(all_graphs)]
            graphs = [all_graphs[i] for i in positions]
            graph_ids = positions
        else:
            positions = [i-1 for i in sorted(selection) if 1 <= i <= len(all_graphs)]
            graphs = [all_graphs[i] for i in positions]
            graph_ids = positions
        
        print(f"  🎯 Processing {len(graphs)} selected graphs")
        
        num_edges = degree * N // 2
        
        if not use_sparse:
            print(f"  🔨 Building dense H_initial matrix {2**N}×{2**N}...", end=" ")
            H_B = build_H_initial(N)
            print("✓")
        else:
            print(f"  🔧 Using SPARSE method (Vectorized Hamiltonians + Brent's Optimization)")
        
        # Process each graph
        for i, (file_pos, edges) in enumerate(zip(graph_ids, graphs)):
            graph_counter += 1
            iter_start = time.time()
            graph_id = file_pos + 1
            
            if use_sparse:
                # SPARSE METHOD with timeout
                result, timed_out = run_with_timeout(
                    find_min_gap_sparse,
                    args=(N, edges, num_edges),
                    kwargs={
                        'target_degeneracy': CONFIG.get('target_degeneracy', 2),
                        's_bounds': CONFIG.get('s_bounds', (0.01, 0.99)),
                        'xtol': CONFIG.get('xtol', 1e-4),
                        'verbose': False
                    },
                    timeout=graph_timeout
                )
                
                if timed_out:
                    timeout_skipped.append(graph_id)
                    print(f"    [{i+1:3d}/{len(graphs)}] Graph_ID={graph_id:3d} | ⏰ TIMEOUT (>{graph_timeout}s)")
                    continue
                
                delta_min, s_min, max_deg, max_cut, num_evals = result
                total_eigsh_calls += num_evals
            else:
                # GRID METHOD (no timeout for legacy method)
                H_P = build_H_problem(N, edges)
                result = find_min_gap_with_degeneracy(H_B, H_P, s_points, num_edges, 
                                                       CONFIG['k_vals_check'], verbose=True)
                delta_min, s_min, max_deg, max_cut, _ = result
                num_evals = 0
            
            # Skip check
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
                'Graph_ID': graph_id,
                'Delta_min': delta_min,
                's_at_min': s_min,
                'Max_degeneracy': max_deg,
                'Max_cut_value': max_cut,
                'Edges': str(edges)
            })
            
            graph_time = time.time() - iter_start
            
            # Progress reporting
            if use_sparse:
                print(f"    [{i+1:3d}/{len(graphs)}] Graph_ID={graph_id:3d} | "
                      f"⏱️  {graph_time:.2f}s | Δ_min={delta_min:.6f} s={s_min:.4f} "
                      f"cut={max_cut} ({num_evals} eigsh)")
            else:
                print(f"    [{i+1:3d}/{len(graphs)}] Graph_ID={graph_id:3d} | "
                      f"⏱️  {graph_time:.1f}s | Δ_min={delta_min:.6f} s={s_min:.2f} "
                      f"cut={max_cut} deg={max_deg}")

    # Save Results
    print("\n" + "-" * 70)
    if len(data) > 0:
        print(f"\n📊 Sorting and saving data...")
        df = pd.DataFrame(data)
        df = df.sort_values(['N', 'Delta_min'], ascending=[True, True])
        df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"💾 Saved to {OUTPUT_FILENAME}")
        
        # Statistics
        total_time = time.time() - start_time
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"   • Total graphs processed: {len(data)}")
        print(f"   • Skipped (degeneracy): {skipped_degeneracy}")
        if timeout_skipped:
            print(f"   • Skipped (timeout): {len(timeout_skipped)}")
            print(f"   • Timed out graph IDs: {timeout_skipped}")
        print(f"   • Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"   • Avg time/graph: {total_time/len(data):.2f}s")
        
        if use_sparse:
            print(f"   • Avg eigsh calls: {total_eigsh_calls/len(data):.1f}")
            
        print(f"\n📈 Overall Statistics:")
        print(f"   • Mean Δ_min: {df['Delta_min'].mean():.6f}")
        print(f"   • Min Δ_min: {df['Delta_min'].min():.6f}")
        print(f"   • Max Δ_min: {df['Delta_min'].max():.6f}")
    else:
        print("\n⚠️  No valid graphs found.")
        if skipped_degeneracy > 0:
            print(f"   • Skipped (degeneracy): {skipped_degeneracy}")
        if timeout_skipped:
            print(f"   • Skipped (timeout): {len(timeout_skipped)}")
            print(f"   • Timed out graph IDs: {timeout_skipped}")
        
    print("=" * 70)

if __name__ == "__main__":
    main()
