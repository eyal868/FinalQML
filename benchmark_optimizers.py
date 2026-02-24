#!/usr/bin/env python3
"""
=========================================================================
Optimizer Benchmark: Compare COBYLA vs SPSA vs L-BFGS-B vs Basin-Hopping
=========================================================================
Benchmarks different classical optimizers for QAOA Max-Cut on a subset
of graphs from the spectral gap dataset.

Usage:
    python benchmark_optimizers.py
    python benchmark_optimizers.py --n-graphs 5 --p-values 1 3 5
    python benchmark_optimizers.py --input outputs/spectral_gap/spectral_gap_3reg_N12_k2.csv
=========================================================================
"""

import numpy as np
import pandas as pd
import time
import ast
import argparse
from typing import List, Tuple, Dict
from scipy.optimize import minimize, dual_annealing

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from output_config import get_run_dirs, save_file, save_run_info
from qaoa_analysis import edges_to_cost_hamiltonian, evaluate_cut_value, calculate_expected_cut

# =========================================================================
# CONFIGURATION
# =========================================================================

DEFAULT_INPUT = 'outputs/spectral_gap/spectral_gap_3reg_N12_k2.csv'
DEFAULT_N_GRAPHS = 10
DEFAULT_P_VALUES = [1, 3, 5, 7]
DEFAULT_MAX_ITER = 500
NUM_SHOTS = 10000
SIMULATOR_METHOD = 'statevector'
RANDOM_SEED = 42


def build_qaoa_cost_fn(edges, n_qubits, p, seed):
    """
    Build and return a QAOA cost function closure.

    Returns:
        (cost_fn, get_result_fn) where:
        - cost_fn(params) -> float (energy to minimize)
        - get_result_fn(optimal_params) -> dict with expected_cut, etc.
    """
    cost_hamiltonian = edges_to_cost_hamiltonian(edges, n_qubits)
    qaoa_circuit = QAOAAnsatz(cost_hamiltonian, reps=p)
    backend = AerSimulator(method=SIMULATOR_METHOD)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled_circuit = pm.run(qaoa_circuit)

    iteration_count = [0]

    def cost_function(params):
        iteration_count[0] += 1
        bound_circuit = transpiled_circuit.assign_parameters(params)
        measured_circuit = bound_circuit.copy()
        measured_circuit.measure_all()

        job = backend.run(measured_circuit, shots=NUM_SHOTS, seed_simulator=seed)
        counts = job.result().get_counts()

        expectation = 0.0
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            bs_rev = bitstring[::-1]
            energy = sum(
                1.0 if bs_rev[u] == bs_rev[v] else -1.0
                for u, v in edges
            )
            expectation += (count / total_shots) * energy
        return expectation

    def get_result(optimal_params):
        bound = transpiled_circuit.assign_parameters(optimal_params)
        mc = bound.copy()
        mc.measure_all()
        job = backend.run(mc, shots=NUM_SHOTS, seed_simulator=seed)
        counts_raw = job.result().get_counts()
        counts_rev = {bs[::-1]: c for bs, c in counts_raw.items()}

        best_bs = max(counts_rev, key=counts_rev.get)
        return {
            'expected_cut': calculate_expected_cut(counts_rev, edges),
            'best_bitstring': best_bs,
            'best_cut_value': evaluate_cut_value(best_bs, edges),
            'num_iterations': iteration_count[0],
        }

    return cost_function, get_result


# =========================================================================
# OPTIMIZER IMPLEMENTATIONS
# =========================================================================

def run_cobyla(cost_fn, initial_params, max_iter):
    """Standard COBYLA (gradient-free)."""
    result = minimize(cost_fn, initial_params, method='COBYLA',
                      options={'maxiter': max_iter})
    return result.x, result.fun


def run_lbfgsb(cost_fn, initial_params, max_iter):
    """L-BFGS-B with finite-difference gradients.
    Bounds parameters to [0, 2π]."""
    n = len(initial_params)
    bounds = [(0, 2 * np.pi)] * n
    result = minimize(cost_fn, initial_params, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': max_iter, 'eps': 0.01})
    return result.x, result.fun


def run_spsa(cost_fn, initial_params, max_iter):
    """Simultaneous Perturbation Stochastic Approximation (SPSA).
    Gradient-free, designed for noisy quantum optimization."""
    params = initial_params.copy()
    n = len(params)

    # SPSA hyperparameters (standard choices)
    a = 0.1
    c = 0.1
    A = max_iter * 0.1
    alpha = 0.602
    gamma = 0.101

    best_params = params.copy()
    best_cost = cost_fn(params)

    for k in range(1, max_iter + 1):
        a_k = a / (k + A) ** alpha
        c_k = c / k ** gamma

        # Random perturbation direction (Bernoulli ±1)
        delta = 2 * np.random.binomial(1, 0.5, size=n) - 1

        # Evaluate at perturbed points
        f_plus = cost_fn(params + c_k * delta)
        f_minus = cost_fn(params - c_k * delta)

        # Gradient estimate
        g_hat = (f_plus - f_minus) / (2 * c_k * delta)

        # Update
        params = params - a_k * g_hat

        # Track best
        if k % 10 == 0 or k == 1:
            current_cost = cost_fn(params)
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()

        # Early stopping if converged
        if k > 50 and abs(f_plus - f_minus) < 1e-6:
            break

    return best_params, best_cost


def run_basin_hopping(cost_fn, initial_params, max_iter):
    """Basin-hopping global optimizer (uses L-BFGS-B as local minimizer)."""
    from scipy.optimize import basinhopping

    n = len(initial_params)
    bounds = [(0, 2 * np.pi)] * n

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {'maxiter': max_iter // 5, 'eps': 0.01}
    }

    result = basinhopping(
        cost_fn, initial_params,
        minimizer_kwargs=minimizer_kwargs,
        niter=5,  # Number of basin-hopping steps
        T=1.0,
        stepsize=0.5,
        seed=RANDOM_SEED
    )
    return result.x, result.fun


OPTIMIZERS = {
    'COBYLA': run_cobyla,
    'L-BFGS-B': run_lbfgsb,
    'SPSA': run_spsa,
    'Basin-Hopping': run_basin_hopping,
}


# =========================================================================
# BENCHMARK RUNNER
# =========================================================================

def benchmark_single(graph_id, edges, n_qubits, optimal_cut, p, optimizer_name, max_iter):
    """Run a single benchmark: one graph, one p, one optimizer."""
    np.random.seed(RANDOM_SEED)
    initial_params = 2 * np.pi * np.random.rand(2 * p)
    seed = RANDOM_SEED + graph_id

    cost_fn, get_result = build_qaoa_cost_fn(edges, n_qubits, p, seed)
    optimizer_fn = OPTIMIZERS[optimizer_name]

    start = time.time()
    try:
        optimal_params, final_cost = optimizer_fn(cost_fn, initial_params.copy(), max_iter)
        elapsed = time.time() - start

        result = get_result(optimal_params)
        approx_ratio = result['expected_cut'] / optimal_cut if optimal_cut > 0 else 0

        return {
            'Graph_ID': graph_id,
            'p': p,
            'optimizer': optimizer_name,
            'approx_ratio': approx_ratio,
            'expected_cut': result['expected_cut'],
            'best_cut': result['best_cut_value'],
            'iterations': result['num_iterations'],
            'final_cost': final_cost,
            'time_s': elapsed,
            'status': 'ok',
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'Graph_ID': graph_id,
            'p': p,
            'optimizer': optimizer_name,
            'approx_ratio': -1,
            'expected_cut': -1,
            'best_cut': -1,
            'iterations': -1,
            'final_cost': np.nan,
            'time_s': elapsed,
            'status': str(e),
        }


def run_benchmark(input_csv, n_graphs, p_values, max_iter, output_csv):
    """Run the full optimizer benchmark."""
    print("=" * 70)
    print("  OPTIMIZER BENCHMARK: COBYLA vs SPSA vs L-BFGS-B vs Basin-Hopping")
    print("=" * 70)

    # Load data
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} graphs from {input_csv}")

    # Sample graphs (spread across gap range)
    df_sorted = df.sort_values('Delta_min')
    indices = np.linspace(0, len(df_sorted) - 1, n_graphs, dtype=int)
    df_sample = df_sorted.iloc[indices].reset_index(drop=True)

    print(f"Selected {n_graphs} graphs spanning Δ_min range "
          f"[{df_sample['Delta_min'].min():.4f}, {df_sample['Delta_min'].max():.4f}]")
    print(f"P values: {p_values}")
    print(f"Optimizers: {list(OPTIMIZERS.keys())}")

    total_runs = n_graphs * len(p_values) * len(OPTIMIZERS)
    print(f"Total benchmark runs: {total_runs}")
    print("-" * 70)

    results = []
    run_count = 0
    total_start = time.time()

    for idx, row in df_sample.iterrows():
        graph_id = row['Graph_ID']
        n_qubits = int(row['N'])
        optimal_cut = row['Max_cut_value']
        edges = ast.literal_eval(row['Edges']) if isinstance(row['Edges'], str) else row['Edges']
        delta_min = row['Delta_min']

        print(f"\nGraph #{graph_id} (N={n_qubits}, Δ_min={delta_min:.4f}, C*={optimal_cut})")

        for p in p_values:
            for opt_name in OPTIMIZERS:
                run_count += 1
                print(f"  [{run_count}/{total_runs}] p={p}, {opt_name}...", end=" ", flush=True)

                result = benchmark_single(graph_id, edges, n_qubits, optimal_cut, p, opt_name, max_iter)
                results.append(result)

                ratio = result['approx_ratio']
                t = result['time_s']
                status = "✓" if ratio > 0 else "✗"
                print(f"{status} ratio={ratio:.4f}, time={t:.1f}s")

    total_time = time.time() - total_start

    # Save results
    results_df = pd.DataFrame(results)
    import os
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to {output_csv}")

    # Desktop mirror copy
    try:
        _, desktop_dir = get_run_dirs("benchmark_optimizers", timestamp=True)
        save_file(output_csv, "benchmark_optimizers", _desktop_dir=desktop_dir)
        save_run_info(desktop_dir, "benchmark_optimizers", extra_info={"n_graphs": n_graphs, "p_values": p_values})
    except Exception as e:
        print(f"  \u26a0\ufe0f Desktop copy skipped: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)

    valid = results_df[results_df['approx_ratio'] > 0]

    for p in p_values:
        print(f"\n  p = {p}:")
        p_data = valid[valid['p'] == p]
        for opt_name in OPTIMIZERS:
            opt_data = p_data[p_data['optimizer'] == opt_name]
            if len(opt_data) > 0:
                mean_ratio = opt_data['approx_ratio'].mean()
                std_ratio = opt_data['approx_ratio'].std()
                mean_time = opt_data['time_s'].mean()
                success = len(opt_data)
                print(f"    {opt_name:15s}: ratio={mean_ratio:.4f}±{std_ratio:.4f}, "
                      f"time={mean_time:.1f}s, success={success}/{n_graphs}")
            else:
                print(f"    {opt_name:15s}: no successful runs")

    # Overall winner per p
    print(f"\n  Best optimizer per p (by mean ratio):")
    for p in p_values:
        p_data = valid[valid['p'] == p]
        if len(p_data) > 0:
            best = p_data.groupby('optimizer')['approx_ratio'].mean().idxmax()
            best_ratio = p_data.groupby('optimizer')['approx_ratio'].mean().max()
            print(f"    p={p}: {best} ({best_ratio:.4f})")

    print(f"\n  Total benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    return results_df


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark QAOA optimizers')
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT,
                        help=f'Input spectral gap CSV (default: {DEFAULT_INPUT})')
    parser.add_argument('--n-graphs', '-n', type=int, default=DEFAULT_N_GRAPHS,
                        help=f'Number of graphs to benchmark (default: {DEFAULT_N_GRAPHS})')
    parser.add_argument('--p-values', '-p', type=int, nargs='+', default=DEFAULT_P_VALUES,
                        help=f'QAOA depths to test (default: {DEFAULT_P_VALUES})')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER,
                        help=f'Max optimizer iterations (default: {DEFAULT_MAX_ITER})')
    parser.add_argument('--output', '-o', type=str, default='outputs/exploratory/optimizer_benchmark.csv',
                        help='Output CSV path')

    args = parser.parse_args()

    run_benchmark(
        input_csv=args.input,
        n_graphs=args.n_graphs,
        p_values=args.p_values,
        max_iter=args.max_iter,
        output_csv=args.output,
    )


if __name__ == '__main__':
    main()
