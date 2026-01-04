#!/usr/bin/env python3
"""
=========================================================================
Weighted Graph Gap Analysis
=========================================================================
Test whether adding random edge weights to 3-regular graphs reduces the 
minimum energy gap in their time-dependent AQC Hamiltonian.

This script:
1. Loads the 10 lowest-gap graphs from filtered QAOA results
2. Applies random weights ~ Uniform(0.1, 2.0) to edges
3. Recalculates the minimum spectral gap for 10 trials per graph
4. Outputs comparison results
=========================================================================
"""

import numpy as np
import pandas as pd
import time
from typing import List, Tuple

from aqc_spectral_utils import (
    build_H_initial_sparse,
    build_H_problem_sparse_weighted,
    get_gap_sparse,
    find_first_gap,
    load_graphs_from_file,
    DEGENERACY_TOL
)
from scipy.optimize import minimize_scalar

# =========================================================================
# CONFIGURATION
# =========================================================================

CONFIG = {
    # Input data
    'input_csv': 'outputs/pipeline_full_N12/QAOA_p_sweep_N12_p1to10_filtered.csv',
    'graph_file': 'graphs_rawdata/12_3_3.scd',
    
    # Analysis parameters
    'num_graphs': 10,           # Number of lowest-gap graphs to analyze
    'num_trials': 10,           # Number of random weight trials per graph
    'weight_min': 0.1,          # Minimum edge weight (exclusive of 0)
    'weight_max': 2.0,          # Maximum edge weight
    'max_weight_attempts': 100, # Max attempts to find valid weights (degeneracy=2)
    
    # Graph parameters
    'N': 12,                    # Number of qubits
    'degree': 3,                # Graph regularity
    
    # Optimization parameters
    's_bounds': (0.01, 0.99),   # Optimization bounds
    'xtol': 1e-4,               # Optimization tolerance
    
    # Output
    'output_csv': 'outputs/weighted_gap_analysis_N12.csv',
    'random_seed': 42,          # For reproducibility (None for random)
}

# =========================================================================
# WEIGHTED GAP COMPUTATION
# =========================================================================

def generate_random_weights(num_edges: int, weight_min: float, weight_max: float) -> List[float]:
    """Generate random weights uniformly distributed in (weight_min, weight_max]."""
    return list(np.random.uniform(weight_min, weight_max, num_edges))


def check_degeneracy_at_s1(H_P_sparse, target_degeneracy: int = 2) -> Tuple[int, float]:
    """
    Check the ground state degeneracy at s=1 (problem Hamiltonian).
    
    Args:
        H_P_sparse: Sparse problem Hamiltonian
        target_degeneracy: Expected degeneracy (default 2 for Max-Cut with Z2 symmetry)
        
    Returns:
        Tuple of (degeneracy, gap_to_first_excited)
    """
    diagonal = H_P_sparse.diagonal()
    k_check = min(target_degeneracy + 2, len(diagonal))  # Check a few extra levels
    evals_s1 = np.partition(diagonal, k_check - 1)[:k_check]
    evals_s1 = np.sort(evals_s1)
    
    gap, degeneracy = find_first_gap(evals_s1, tol=DEGENERACY_TOL)
    return degeneracy, gap


def generate_valid_weights(
    N: int,
    edges: List[Tuple[int, int]],
    num_edges: int,
    weight_min: float,
    weight_max: float,
    max_attempts: int = 100,
    target_degeneracy: int = 2
) -> Tuple[List[float], int, int]:
    """
    Generate random weights that maintain degeneracy=2 at s=1.
    
    Some weight combinations can accidentally create higher degeneracy
    (different cuts with same weighted energy). This function rejects
    such weights and regenerates until valid weights are found.
    
    Args:
        N: Number of qubits
        edges: List of edges
        num_edges: Number of edges
        weight_min: Minimum weight value
        weight_max: Maximum weight value
        max_attempts: Maximum attempts before giving up
        target_degeneracy: Required degeneracy at s=1 (default 2)
        
    Returns:
        Tuple of (valid_weights, degeneracy_found, num_attempts)
        Returns (None, degeneracy_found, max_attempts) if no valid weights found
    """
    for attempt in range(1, max_attempts + 1):
        weights = generate_random_weights(num_edges, weight_min, weight_max)
        H_P_sparse = build_H_problem_sparse_weighted(N, edges, weights)
        degeneracy, _ = check_degeneracy_at_s1(H_P_sparse, target_degeneracy)
        
        if degeneracy == target_degeneracy:
            return weights, degeneracy, attempt
    
    # Failed to find valid weights after max_attempts
    return None, degeneracy, max_attempts


def find_min_gap_weighted(
    N: int,
    edges: List[Tuple[int, int]],
    weights: List[float],
    s_bounds: Tuple[float, float] = (0.01, 0.99),
    xtol: float = 1e-4
) -> Tuple[float, float, int]:
    """
    Find minimum spectral gap for a weighted graph.
    
    Args:
        N: Number of qubits
        edges: List of edges as (i, j) tuples
        weights: List of weights for each edge
        s_bounds: Optimization bounds for s parameter
        xtol: Optimization tolerance
        
    Returns:
        Tuple of (min_gap, s_at_min, num_function_evals)
    """
    # Build sparse Hamiltonians
    H_B_sparse = build_H_initial_sparse(N)
    H_P_sparse = build_H_problem_sparse_weighted(N, edges, weights)
    
    # Track function evaluations
    eval_counter = [0]
    
    def gap_function(s: float) -> float:
        """Gap function to minimize: Δ(s) = E_2(s) - E_0(s)"""
        eval_counter[0] += 1
        return get_gap_sparse(s, H_B_sparse, H_P_sparse, k_eigenvalues=3)
    
    # Use Brent's method for bounded scalar optimization
    result = minimize_scalar(
        gap_function,
        bounds=s_bounds,
        method='bounded',
        options={'xatol': xtol}
    )
    
    return float(result.fun), float(result.x), eval_counter[0]


# =========================================================================
# MAIN ANALYSIS
# =========================================================================

def main():
    """Main execution: analyze effect of random edge weights on spectral gap."""
    print("=" * 70)
    print("  WEIGHTED GRAPH GAP ANALYSIS")
    print("=" * 70)
    
    # Set random seed for reproducibility
    if CONFIG['random_seed'] is not None:
        np.random.seed(CONFIG['random_seed'])
        print(f"\n🎲 Random seed: {CONFIG['random_seed']}")
    
    print(f"\n📊 Configuration:")
    print(f"  • Input CSV: {CONFIG['input_csv']}")
    print(f"  • Graph file: {CONFIG['graph_file']}")
    print(f"  • Number of graphs: {CONFIG['num_graphs']}")
    print(f"  • Trials per graph: {CONFIG['num_trials']}")
    print(f"  • Weight range: ({CONFIG['weight_min']}, {CONFIG['weight_max']}]")
    print(f"  • Max weight attempts: {CONFIG['max_weight_attempts']} (to ensure degeneracy=2)")
    print(f"  • Output: {CONFIG['output_csv']}")
    
    # Load QAOA results
    print(f"\n📖 Loading QAOA results...")
    df_qaoa = pd.read_csv(CONFIG['input_csv'])
    df_qaoa = df_qaoa.sort_values('Delta_min', ascending=True)
    
    # Select lowest-gap graphs
    selected_graphs = df_qaoa.head(CONFIG['num_graphs'])
    print(f"  ✓ Selected {len(selected_graphs)} lowest-gap graphs")
    print(f"  • Gap range: {selected_graphs['Delta_min'].min():.4f} - {selected_graphs['Delta_min'].max():.4f}")
    
    # Load graph data
    print(f"\n📖 Loading graph data from {CONFIG['graph_file']}...")
    all_graphs = load_graphs_from_file(CONFIG['graph_file'])
    print(f"  ✓ Loaded {len(all_graphs)} graphs")
    
    # Prepare results storage
    results = []
    N = CONFIG['N']
    num_edges = CONFIG['degree'] * N // 2
    total_weight_rejections = 0
    failed_trials = 0
    
    start_time = time.time()
    print(f"\n🚀 Starting analysis...")
    print(f"   Note: Only weights with degeneracy=2 at s=1 are accepted")
    print("-" * 70)
    
    # Process each selected graph
    for idx, row in selected_graphs.iterrows():
        graph_id = int(row['Graph_ID'])
        original_gap = row['Delta_min']
        
        # Get edges (Graph_ID is 1-indexed)
        edges = all_graphs[graph_id - 1]
        
        print(f"\n▶ Graph {graph_id} (Original Δ_min = {original_gap:.6f})")
        
        trial_gaps = []
        graph_rejections = 0
        
        for trial in range(CONFIG['num_trials']):
            trial_start = time.time()
            
            # Generate valid random weights (degeneracy=2 at s=1)
            weights, degeneracy, attempts = generate_valid_weights(
                N, edges, num_edges,
                CONFIG['weight_min'], 
                CONFIG['weight_max'],
                max_attempts=CONFIG['max_weight_attempts']
            )
            
            rejections = attempts - 1
            graph_rejections += rejections
            total_weight_rejections += rejections
            
            if weights is None:
                # Failed to find valid weights
                failed_trials += 1
                print(f"    Trial {trial+1:2d}: ⚠️ FAILED - Could not find weights with degeneracy=2 "
                      f"(last deg={degeneracy}, {attempts} attempts)")
                continue
            
            # Find minimum gap for weighted graph
            weighted_gap, s_at_min, num_evals = find_min_gap_weighted(
                N, edges, weights,
                s_bounds=CONFIG['s_bounds'],
                xtol=CONFIG['xtol']
            )
            
            trial_time = time.time() - trial_start
            trial_gaps.append(weighted_gap)
            
            # Calculate change
            gap_change_pct = ((weighted_gap - original_gap) / original_gap) * 100
            
            # Store result
            results.append({
                'Graph_ID': graph_id,
                'Original_Delta_min': original_gap,
                'Trial': trial + 1,
                'Weights': str(weights),
                'Weighted_Delta_min': weighted_gap,
                's_at_min': s_at_min,
                'Gap_Change_Percent': gap_change_pct,
                'Num_Evals': num_evals,
                'Weight_Attempts': attempts,
                'Trial_Time_s': trial_time
            })
            
            # Progress indicator
            change_symbol = "↓" if gap_change_pct < 0 else "↑" if gap_change_pct > 0 else "="
            attempt_str = f" [{attempts} att]" if attempts > 1 else ""
            print(f"    Trial {trial+1:2d}: Δ_min = {weighted_gap:.6f} "
                  f"({change_symbol} {abs(gap_change_pct):5.1f}%){attempt_str} | {trial_time:.2f}s")
        
        # Summary for this graph
        if trial_gaps:
            mean_weighted = np.mean(trial_gaps)
            std_weighted = np.std(trial_gaps)
            min_weighted = np.min(trial_gaps)
            max_weighted = np.max(trial_gaps)
            
            reject_str = f", {graph_rejections} weight rejections" if graph_rejections > 0 else ""
            print(f"  📊 Summary: mean={mean_weighted:.6f} (±{std_weighted:.6f}), "
                  f"range=[{min_weighted:.6f}, {max_weighted:.6f}]{reject_str}")
        else:
            print(f"  ⚠️ No valid trials for this graph")
    
    # Save results
    print("\n" + "-" * 70)
    df_results = pd.DataFrame(results)
    df_results.to_csv(CONFIG['output_csv'], index=False)
    print(f"\n💾 Results saved to {CONFIG['output_csv']}")
    
    # Overall summary
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 70}")
    print("  OVERALL SUMMARY")
    print("=" * 70)
    
    # Aggregate statistics
    if len(df_results) > 0:
        avg_change = df_results['Gap_Change_Percent'].mean()
        improvements = (df_results['Gap_Change_Percent'] < 0).sum()
        total_valid_trials = len(df_results)
        
        print(f"\n📈 Statistics across all {total_valid_trials} valid trials:")
        print(f"  • Average gap change: {avg_change:+.2f}%")
        print(f"  • Trials with reduced gap: {improvements}/{total_valid_trials} ({100*improvements/total_valid_trials:.1f}%)")
        print(f"  • Trials with increased gap: {total_valid_trials - improvements}/{total_valid_trials} ({100*(total_valid_trials-improvements)/total_valid_trials:.1f}%)")
        
        # Degeneracy check statistics
        print(f"\n🔍 Degeneracy validation:")
        print(f"  • Total weight rejections (deg≠2): {total_weight_rejections}")
        print(f"  • Failed trials (no valid weights found): {failed_trials}")
        if 'Weight_Attempts' in df_results.columns:
            avg_attempts = df_results['Weight_Attempts'].mean()
            max_attempts = df_results['Weight_Attempts'].max()
            print(f"  • Average attempts per valid trial: {avg_attempts:.1f}")
            print(f"  • Max attempts for a trial: {max_attempts}")
    else:
        print(f"\n⚠️ No valid trials completed!")
        print(f"  • Total weight rejections: {total_weight_rejections}")
        print(f"  • Failed trials: {failed_trials}")
    
    # Per-graph summary
    if len(df_results) > 0:
        print(f"\n📊 Per-graph summary:")
        for graph_id in selected_graphs['Graph_ID'].values:
            graph_data = df_results[df_results['Graph_ID'] == graph_id]
            if len(graph_data) > 0:
                orig = graph_data['Original_Delta_min'].iloc[0]
                mean_weighted = graph_data['Weighted_Delta_min'].mean()
                improvements_graph = (graph_data['Gap_Change_Percent'] < 0).sum()
                avg_change_graph = graph_data['Gap_Change_Percent'].mean()
                valid_trials = len(graph_data)
                
                print(f"  Graph {graph_id:3d}: orig={orig:.4f}, weighted_mean={mean_weighted:.4f}, "
                      f"change={avg_change_graph:+.1f}%, improved={improvements_graph}/{valid_trials}")
            else:
                print(f"  Graph {graph_id:3d}: No valid trials")
    
    print(f"\n⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print("=" * 70)


if __name__ == "__main__":
    main()

