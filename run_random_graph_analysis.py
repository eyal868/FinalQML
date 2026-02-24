#!/usr/bin/env python3
"""
=========================================================================
Random Graph Analysis: Combined Spectral Gap + QAOA Pipeline
=========================================================================
Standalone script for analyzing QAOA performance on random graphs.

Workflow:
1. Generate random graphs (Erdős-Rényi, random regular, etc.)
2. Compute spectral gaps (unweighted and optionally weighted)
3. Run QAOA p-sweeps with success probability + approximation ratio
4. Save results in standard CSV format compatible with all plotting tools

Usage:
    python run_random_graph_analysis.py                         # Defaults
    python run_random_graph_analysis.py --model erdos_renyi --N 10 --num-graphs 20
    python run_random_graph_analysis.py --model random_regular --degree 3 --N 12
    python run_random_graph_analysis.py --weighted              # Add random weights
    python run_random_graph_analysis.py --skip-qaoa             # Only spectral gaps
=========================================================================
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Tuple

from graph_generation import (
    generate_random_graphs_batch,
    compute_spectral_gaps_for_graphs
)
from aqc_spectral_utils import (
    build_H_problem_sparse_weighted,
    compute_weighted_optimal_cut
)
from weighted_gap_analysis import generate_valid_weights
from output_config import get_run_dirs, save_file, save_run_info


# =========================================================================
# ARGUMENT PARSER
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Random Graph Analysis: Spectral gaps + QAOA on random graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Graph generation
    parser.add_argument('--model', type=str, default='erdos_renyi',
                        choices=['erdos_renyi', 'random_regular', 'watts_strogatz'],
                        help='Random graph model (default: erdos_renyi)')
    parser.add_argument('--N', type=int, default=10,
                        help='Number of vertices (default: 10)')
    parser.add_argument('--num-graphs', type=int, default=20,
                        help='Number of graphs to generate (default: 20)')
    parser.add_argument('--edge-prob', type=float, default=0.5,
                        help='Edge probability for Erdos-Renyi (default: 0.5)')
    parser.add_argument('--degree', type=int, default=3,
                        help='Degree for random regular graphs (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')

    # Spectral gap options
    parser.add_argument('--degeneracy', type=int, default=None,
                        help='Filter to specific degeneracy (default: accept all)')

    # Weighted analysis
    parser.add_argument('--weighted', action='store_true',
                        help='Also run weighted analysis (random edge weights)')
    parser.add_argument('--num-weight-trials', type=int, default=5,
                        help='Number of random weight trials per graph (default: 5)')
    parser.add_argument('--weight-min', type=float, default=0.1,
                        help='Minimum edge weight (default: 0.1)')
    parser.add_argument('--weight-max', type=float, default=2.0,
                        help='Maximum edge weight (default: 2.0)')

    # QAOA options
    parser.add_argument('--skip-qaoa', action='store_true',
                        help='Skip QAOA analysis (only compute spectral gaps)')
    parser.add_argument('--p-min', type=int, default=1,
                        help='Minimum QAOA depth (default: 1)')
    parser.add_argument('--p-max', type=int, default=5,
                        help='Maximum QAOA depth (default: 5)')
    parser.add_argument('--max-iter', type=int, default=500,
                        help='Max optimizer iterations (default: 500)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputs/qaoa_random/)')

    return parser.parse_args()


# =========================================================================
# MAIN
# =========================================================================

def main():
    args = parse_args()
    N = args.N
    model = args.model

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f'outputs/qaoa_random/{model}_N{N}/'

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  RANDOM GRAPH ANALYSIS")
    print("=" * 70)
    print(f"  Model:         {model}")
    print(f"  N:             {N}")
    print(f"  Num graphs:    {args.num_graphs}")
    print(f"  Weighted:      {args.weighted}")
    print(f"  QAOA:          {'skip' if args.skip_qaoa else f'p={args.p_min}-{args.p_max}'}")
    print(f"  Output dir:    {args.output_dir}")

    total_start = time.time()

    # =====================================================================
    # Step 1: Generate random graphs
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 1: GENERATING {args.num_graphs} RANDOM GRAPHS")
    print(f"{'='*70}")

    params = {}
    if model == 'erdos_renyi':
        params['edge_prob'] = args.edge_prob
        print(f"  Edge probability: {args.edge_prob}")
    elif model == 'random_regular':
        params['degree'] = args.degree
        print(f"  Degree: {args.degree}")
    elif model == 'watts_strogatz':
        params['k'] = args.degree
        params['rewire_prob'] = args.edge_prob
        print(f"  k={args.degree}, rewire_prob={args.edge_prob}")

    graphs = generate_random_graphs_batch(
        N=N,
        num_graphs=args.num_graphs,
        model=model,
        base_seed=args.seed,
        **params
    )
    print(f"  Generated {len(graphs)} connected graphs")

    # =====================================================================
    # Step 2: Compute spectral gaps
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 2: COMPUTING SPECTRAL GAPS")
    print(f"{'='*70}")

    df_gaps = compute_spectral_gaps_for_graphs(
        N=N,
        graphs=graphs,
        target_degeneracy=args.degeneracy,
        verbose=True
    )

    # Add model info
    df_gaps['Graph_Model'] = model

    gap_csv = os.path.join(args.output_dir, f"spectral_gap_{model}_N{N}.csv")
    df_gaps.to_csv(gap_csv, index=False)
    print(f"\n  Saved {len(df_gaps)} spectral gaps to {gap_csv}")

    if len(df_gaps) > 0:
        print(f"  Gap range: [{df_gaps['Delta_min'].min():.4f}, {df_gaps['Delta_min'].max():.4f}]")
        print(f"  Mean gap:  {df_gaps['Delta_min'].mean():.4f}")

    # =====================================================================
    # Step 3: Weighted analysis (optional)
    # =====================================================================
    if args.weighted and len(df_gaps) > 0:
        print(f"\n{'='*70}")
        print(f"  STEP 3: WEIGHTED SPECTRAL GAP ANALYSIS")
        print(f"{'='*70}")

        weighted_results = []
        import ast

        for _, row in df_gaps.iterrows():
            graph_id = row['Graph_ID']
            edges = ast.literal_eval(row['Edges'])

            for trial in range(args.num_weight_trials):
                weights, deg, attempts = generate_valid_weights(
                    N=N,
                    edges=edges,
                    num_edges=len(edges),
                    weight_min=args.weight_min,
                    weight_max=args.weight_max,
                    target_degeneracy=row['Max_degeneracy']
                )

                if weights is None:
                    continue

                # Compute weighted gap by reusing compute_spectral_gaps_for_graphs
                # with a single graph + weights
                df_w = compute_spectral_gaps_for_graphs(
                    N=N,
                    graphs=[edges],
                    weights_list=[weights],
                    verbose=False,
                    graph_id_offset=graph_id
                )

                if len(df_w) > 0:
                    w_row = df_w.iloc[0].to_dict()
                    w_row['Trial'] = trial + 1
                    w_row['Original_Delta_min'] = row['Delta_min']
                    w_row['Weighted_Delta_min'] = w_row['Delta_min']
                    w_row['Gap_Change_Percent'] = (
                        (w_row['Delta_min'] - row['Delta_min']) / row['Delta_min'] * 100
                        if row['Delta_min'] > 0 else 0
                    )
                    w_row['Weights'] = str(weights)
                    w_row['Graph_Model'] = model
                    weighted_results.append(w_row)

            print(f"  Graph {graph_id}: {len([r for r in weighted_results if r['Graph_ID'] == graph_id])} valid weight trials")

        if weighted_results:
            df_weighted = pd.DataFrame(weighted_results)
            weighted_csv = os.path.join(args.output_dir, f"weighted_gap_{model}_N{N}.csv")
            df_weighted.to_csv(weighted_csv, index=False)
            print(f"\n  Saved {len(df_weighted)} weighted trials to {weighted_csv}")
            print(f"  Weighted gap range: [{df_weighted['Weighted_Delta_min'].min():.4f}, {df_weighted['Weighted_Delta_min'].max():.4f}]")

    # =====================================================================
    # Step 4: QAOA analysis (optional)
    # =====================================================================
    if not args.skip_qaoa and len(df_gaps) > 0:
        print(f"\n{'='*70}")
        print(f"  STEP 4: QAOA P-SWEEP ANALYSIS")
        print(f"{'='*70}")

        import ast
        from qaoa_analysis import run_qaoa, calculate_success_probability
        import qaoa_analysis as qa

        # Configure QAOA module
        qa.MAX_OPTIMIZER_ITERATIONS = args.max_iter
        qa.RANDOM_SEED = args.seed

        p_values = list(range(args.p_min, args.p_max + 1))
        qaoa_results = []

        for idx, (_, row) in enumerate(df_gaps.iterrows()):
            graph_id = row['Graph_ID']
            edges = ast.literal_eval(row['Edges'])
            optimal_cut = row['Max_cut_value']

            print(f"  [{idx+1}/{len(df_gaps)}] Graph {graph_id} (gap={row['Delta_min']:.4f}, opt_cut={optimal_cut})")

            result_row = {
                'N': N,
                'Graph_ID': graph_id,
                'Delta_min': row['Delta_min'],
                's_at_min': row['s_at_min'],
                'Max_degeneracy': row['Max_degeneracy'],
                'Optimal_cut': optimal_cut,
                'Edges': row['Edges'],
                'Graph_Model': model,
            }

            previous_optimal_params = None
            for p in p_values:
                try:
                    p_start = time.time()

                    # Warm-start
                    initial_params = None
                    if previous_optimal_params is not None:
                        from qaoa_analysis import extend_params_for_warmstart
                        initial_params = extend_params_for_warmstart(previous_optimal_params, p)
                    elif p == 1:
                        from qaoa_analysis import get_heuristic_initial_params
                        initial_params = get_heuristic_initial_params(p)

                    qaoa_result = run_qaoa(
                        edges=edges,
                        n_qubits=N,
                        p=p,
                        max_iter=args.max_iter,
                        initial_params=initial_params,
                        optimal_cut=optimal_cut
                    )

                    previous_optimal_params = qaoa_result['optimal_params']
                    p_time = time.time() - p_start

                    expected_cut = qaoa_result['expected_cut']
                    approx_ratio = expected_cut / optimal_cut if optimal_cut > 0 else 0
                    success_prob = qaoa_result.get('success_prob', -1)

                    result_row[f'p{p}_expected_cut'] = expected_cut
                    result_row[f'p{p}_approx_ratio'] = approx_ratio
                    result_row[f'p{p}_success_prob'] = success_prob if success_prob is not None else -1
                    result_row[f'p{p}_most_prob_cut'] = qaoa_result['most_probable_cut']
                    result_row[f'p{p}_most_prob_prob'] = qaoa_result['most_probable_prob']
                    result_row[f'p{p}_iterations'] = qaoa_result['num_iterations']
                    result_row[f'p{p}_time'] = p_time

                except Exception as e:
                    print(f"    Error at p={p}: {e}")
                    result_row[f'p{p}_expected_cut'] = -1
                    result_row[f'p{p}_approx_ratio'] = -1
                    result_row[f'p{p}_success_prob'] = -1
                    result_row[f'p{p}_most_prob_cut'] = -1
                    result_row[f'p{p}_most_prob_prob'] = -1
                    result_row[f'p{p}_iterations'] = -1
                    result_row[f'p{p}_time'] = -1

            qaoa_results.append(result_row)

            # Print summary for this graph
            best_p = max(p_values, key=lambda p: result_row.get(f'p{p}_approx_ratio', -1))
            print(f"    Best: p={best_p}, ratio={result_row[f'p{best_p}_approx_ratio']:.4f}, "
                  f"success_prob={result_row[f'p{best_p}_success_prob']:.4f}")

        # Save QAOA results
        df_qaoa = pd.DataFrame(qaoa_results)
        p_range = f"p{args.p_min}to{args.p_max}"
        qaoa_csv = os.path.join(args.output_dir, f"qaoa_sweep_{model}_N{N}_{p_range}.csv")
        df_qaoa.to_csv(qaoa_csv, index=False)
        print(f"\n  Saved QAOA results to {qaoa_csv}")

        # Apply monotonic filter
        from filter_qaoa_monotonic import process_file
        process_file(qaoa_csv, verbose=True)

    # =====================================================================
    # Summary
    # =====================================================================
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
