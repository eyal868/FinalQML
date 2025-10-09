#!/usr/bin/env python3
"""
=========================================================================
QAOA Performance Analysis for Max-Cut Problem
=========================================================================
Research Goal: Analyze QAOA performance on graphs and correlate with 
their spectral gap (Δ_min) values.

This script runs QAOA on the same graphs analyzed in spectral_gap_analysis.py
to test the hypothesis: graphs with smaller Δ_min are harder for QAOA.

Metrics tracked:
- Approximation ratio (QAOA_cut / optimal_cut)
- Number of optimizer iterations to converge
- Final cost function value
=========================================================================
"""

import numpy as np
import pandas as pd
import time
import ast
from typing import List, Tuple, Dict
from scipy.optimize import minimize

# Qiskit imports (following IBM tutorial)
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# =========================================================================
# 1. CONFIGURATION PARAMETERS
# =========================================================================

# Input data file (from spectral gap analysis)
INPUT_CSV = 'outputs/Delta_min_3_regular_N12_res20.csv'

# QAOA parameters - p-sweep mode
P_VALUES_TO_TEST = list(range(1, 11))  # Test p=1,2,3,...,10
MAX_OPTIMIZER_ITERATIONS = 500  # Maximum classical optimizer iterations
OPTIMIZER_METHOD = 'COBYLA'     # Classical optimizer (COBYLA from tutorial)
NUM_SHOTS = 10000               # Number of measurement shots

# Output filename
OUTPUT_FILENAME = 'outputs/QAOA_p_sweep_N12_p1to10.csv'

# Simulation backend
SIMULATOR_METHOD = 'statevector'  # Noiseless simulation

# Random seed for reproducibility
RANDOM_SEED = 42

# =========================================================================
# 2. HELPER FUNCTIONS: BUILD COST HAMILTONIAN
# =========================================================================

def edges_to_cost_hamiltonian(edges: List[Tuple[int, int]], n_qubits: int) -> SparsePauliOp:
    """
    Constructs the Max-Cut cost Hamiltonian as a SparsePauliOp.
    
    For Max-Cut: H_cost = ∑_{(i,j)∈E} Z_i Z_j
    
    Args:
        edges: List of edges as (vertex1, vertex2) tuples
        n_qubits: Number of qubits (nodes in graph)
        
    Returns:
        SparsePauliOp representing the cost Hamiltonian
    """
    # Build list of Pauli strings and coefficients
    pauli_list = []
    
    for u, v in edges:
        # Create Pauli string: Z on qubits u and v, I elsewhere
        pauli_str = ['I'] * n_qubits
        pauli_str[u] = 'Z'
        pauli_str[v] = 'Z'
        
        # Qiskit uses little-endian: reverse the string
        pauli_str_reversed = ''.join(pauli_str[::-1])
        
        # Add to list with coefficient +1.0
        pauli_list.append((pauli_str_reversed, 1.0))
    
    # Create SparsePauliOp from list
    cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
    
    return cost_hamiltonian


def evaluate_cut_value(bitstring: str, edges: List[Tuple[int, int]]) -> int:
    """
    Evaluate the cut value for a given bitstring assignment.
    
    Args:
        bitstring: Binary string (e.g., '01010')
        edges: List of edges
        
    Returns:
        Number of edges in the cut
    """
    cut_value = 0
    for u, v in edges:
        # Edge is in cut if endpoints have different values
        if bitstring[u] != bitstring[v]:
            cut_value += 1
    return cut_value


def calculate_expected_cut(counts: Dict[str, int], edges: List[Tuple[int, int]]) -> float:
    """
    Calculate expected cut value from measurement counts.
    
    Expected cut = ∑_bitstrings P(bitstring) × cut_value(bitstring)
    
    Args:
        counts: Dictionary mapping bitstrings to counts
        edges: List of edges
        
    Returns:
        Expected cut value (as float)
    """
    expected_cut = 0.0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        prob = count / total_shots
        cut_val = evaluate_cut_value(bitstring, edges)
        expected_cut += prob * cut_val
    
    return expected_cut


# =========================================================================
# 3. QAOA OPTIMIZATION FUNCTION
# =========================================================================

def run_qaoa(edges: List[Tuple[int, int]], 
             n_qubits: int,
             p: int = 1,
             max_iter: int = 200,
             initial_params: np.ndarray = None) -> Dict:
    """
    Run QAOA on a Max-Cut problem instance.
    
    Args:
        edges: Graph edges as list of tuples
        n_qubits: Number of qubits (nodes)
        p: Number of QAOA layers
        max_iter: Maximum optimizer iterations
        initial_params: Initial parameter values (random if None)
        
    Returns:
        Dictionary with results:
        - 'best_bitstring': Best solution found
        - 'best_cut_value': Cut value of best solution
        - 'num_iterations': Number of optimizer iterations
        - 'final_cost': Final cost function value
        - 'optimization_time': Time spent in optimization (seconds)
        - 'optimal_params': Optimal parameters found
    """
    # Build cost Hamiltonian
    cost_hamiltonian = edges_to_cost_hamiltonian(edges, n_qubits)
    
    # Create QAOA ansatz circuit
    qaoa_circuit = QAOAAnsatz(cost_hamiltonian, reps=p)
    
    # Setup simulator
    backend = AerSimulator(method=SIMULATOR_METHOD)
    
    # Transpile circuit for backend
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled_circuit = pm.run(qaoa_circuit)
    
    # Counter for iterations
    iteration_count = [0]
    cost_history = []
    
    # Cost function to minimize
    def cost_function(params):
        """Evaluate expectation value of cost Hamiltonian."""
        iteration_count[0] += 1
        
        # Bind parameters to circuit
        bound_circuit = transpiled_circuit.assign_parameters(params)
        
        # Add measurements to the circuit for sampling
        from qiskit import QuantumCircuit
        measured_circuit = bound_circuit.copy()
        measured_circuit.measure_all()
        
        # Run circuit and get counts
        job = backend.run(measured_circuit, shots=NUM_SHOTS, seed_simulator=RANDOM_SEED)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation value
        expectation = 0.0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Reverse bitstring (Qiskit little-endian)
            bitstring_reversed = bitstring[::-1]
            
            # Calculate energy for this bitstring
            energy = 0.0
            for u, v in edges:
                # Z_i Z_j eigenvalue: +1 if same, -1 if different
                if bitstring_reversed[u] == bitstring_reversed[v]:
                    energy += 1.0
                else:
                    energy -= 1.0
            
            expectation += (count / total_shots) * energy
        
        cost_history.append(expectation)
        
        # Print progress every 10 iterations
        if iteration_count[0] % 10 == 0:
            print(f"      Iteration {iteration_count[0]:3d}: Cost = {expectation:.6f}")
        
        return expectation
    
    # Initial parameters (random if not provided)
    if initial_params is None:
        np.random.seed(RANDOM_SEED)
        initial_params = 2 * np.pi * np.random.rand(2 * p)
    
    # Run classical optimization
    print(f"    Starting QAOA optimization (p={p}, max_iter={max_iter})...")
    start_time = time.time()
    
    result = minimize(
        cost_function,
        initial_params,
        method=OPTIMIZER_METHOD,
        options={'maxiter': max_iter}
    )
    
    optimization_time = time.time() - start_time
    
    # Get final solution
    optimal_params = result.x
    
    # Sample from final circuit to get best bitstring
    from qiskit import QuantumCircuit
    bound_circuit_final = transpiled_circuit.assign_parameters(optimal_params)
    measured_circuit_final = bound_circuit_final.copy()
    measured_circuit_final.measure_all()
    
    job_final = backend.run(measured_circuit_final, shots=NUM_SHOTS, seed_simulator=RANDOM_SEED)
    result_final = job_final.result()
    counts_final = result_final.get_counts()
    
    # Convert all bitstrings to big-endian (reverse Qiskit's little-endian)
    counts_reversed = {}
    for bitstring, count in counts_final.items():
        counts_reversed[bitstring[::-1]] = count
    
    # Find most probable bitstring
    most_probable_bitstring = max(counts_reversed, key=counts_reversed.get)
    most_probable_count = counts_reversed[most_probable_bitstring]
    most_probable_prob = most_probable_count / NUM_SHOTS
    most_probable_cut = evaluate_cut_value(most_probable_bitstring, edges)
    
    # Calculate expected cut value (correct approximation ratio)
    expected_cut = calculate_expected_cut(counts_reversed, edges)
    
    # Best measured bitstring (for reference)
    best_bitstring = most_probable_bitstring
    best_cut_value = most_probable_cut
    
    return {
        'best_bitstring': best_bitstring,
        'best_cut_value': best_cut_value,
        'most_probable_bitstring': most_probable_bitstring,
        'most_probable_prob': most_probable_prob,
        'most_probable_cut': most_probable_cut,
        'expected_cut': expected_cut,
        'num_iterations': iteration_count[0],
        'final_cost': result.fun,
        'optimization_time': optimization_time,
        'optimal_params': optimal_params,
        'cost_history': cost_history
    }


# =========================================================================
# 4. EXAMPLE: 3-NODE TRIANGLE (VALIDATION)
# =========================================================================

def run_3node_example():
    """
    Run QAOA on simple 3-node triangle graph to validate implementation.
    
    Graph: 0 -- 1
           \\  /
            2
    
    Optimal cut: 2 edges (e.g., {0} vs {1,2})
    """
    print("=" * 70)
    print("  QAOA VALIDATION: 3-NODE TRIANGLE")
    print("=" * 70)
    
    # Define 3-node triangle
    edges = [(0, 1), (1, 2), (0, 2)]
    n_qubits = 3
    optimal_cut = 2  # Known optimal value
    
    print(f"\nGraph: {edges}")
    print(f"Optimal cut value: {optimal_cut}")
    print(f"Configuration: p={P_LAYERS}, max_iter={MAX_OPTIMIZER_ITERATIONS}, shots={NUM_SHOTS}")
    
    # Run QAOA
    result = run_qaoa(edges, n_qubits, p=P_LAYERS, max_iter=MAX_OPTIMIZER_ITERATIONS)
    
    # Calculate approximation ratio
    approx_ratio = result['best_cut_value'] / optimal_cut
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Best bitstring: {result['best_bitstring']}")
    print(f"  Best cut value: {result['best_cut_value']}")
    print(f"  Optimal cut value: {optimal_cut}")
    print(f"  Approximation ratio: {approx_ratio:.6f}")
    print(f"  Optimizer iterations: {result['num_iterations']}")
    print(f"  Final cost: {result['final_cost']:.6f}")
    print(f"  Optimization time: {result['optimization_time']:.2f}s")
    print(f"{'='*70}\n")
    
    return result


# =========================================================================
# 5. MAIN: ANALYZE GRAPHS FROM SPECTRAL GAP DATA
# =========================================================================

def analyze_graphs_from_csv(csv_path: str = INPUT_CSV):
    """
    Load graphs from spectral gap analysis CSV and run QAOA p-sweep on each.
    
    Tests multiple p values (1 through 10) for each graph and saves all results.
    
    Args:
        csv_path: Path to CSV file with spectral gap data
    """
    print("=" * 70)
    print("  QAOA P-SWEEP ANALYSIS ON SPECTRAL GAP GRAPHS")
    print("=" * 70)
    
    # Load data
    print(f"\n📖 Loading graph data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} graphs for N={df['N'].iloc[0]}")
    
    print(f"\n📊 Configuration:")
    print(f"   • QAOA layers to test: p={P_VALUES_TO_TEST}")
    print(f"   • Max optimizer iterations: {MAX_OPTIMIZER_ITERATIONS}")
    print(f"   • Optimizer method: {OPTIMIZER_METHOD}")
    print(f"   • Number of shots: {NUM_SHOTS}")
    print(f"   • Simulator method: {SIMULATOR_METHOD}")
    print(f"   • Random seed: {RANDOM_SEED}")
    
    # Prepare results storage
    results_data = []
    
    print(f"\n🚀 Starting QAOA analysis on {len(df)} graphs...")
    print("-" * 70)
    
    total_start_time = time.time()
    
    # Process each graph
    for idx, row in df.iterrows():
        graph_id = row['Graph_ID']
        n_qubits = row['N']
        delta_min = row['Delta_min']
        optimal_cut = row['Max_cut_value']
        edges = ast.literal_eval(row['Edges'])
        
        print(f"\n[{idx+1}/{len(df)}] Graph #{graph_id} | N={n_qubits} | Δ_min={delta_min:.6f} | Optimal_cut={optimal_cut}")
        
        # Run QAOA for all p values
        try:
            # Storage for this graph's results across all p
            p_results = {
                'N': n_qubits,
                'Graph_ID': graph_id,
                'Delta_min': delta_min,
                's_at_min': row['s_at_min'],
                'Max_degeneracy': row['Max_degeneracy'],
                'Optimal_cut': optimal_cut,
            }
            
            # Test each p value
            for p in P_VALUES_TO_TEST:
                print(f"    Testing p={p}...", end=" ")
                p_start = time.time()
                
                qaoa_result = run_qaoa(
                    edges=edges,
                    n_qubits=n_qubits,
                    p=p,
                    max_iter=MAX_OPTIMIZER_ITERATIONS
                )
                
                p_time = time.time() - p_start
                
                # Calculate approximation ratio (using expected cut)
                expected_cut = qaoa_result['expected_cut']
                approx_ratio = expected_cut / optimal_cut
                
                # Store results for this p
                p_results[f'p{p}_expected_cut'] = expected_cut
                p_results[f'p{p}_approx_ratio'] = approx_ratio
                p_results[f'p{p}_most_prob_cut'] = qaoa_result['most_probable_cut']
                p_results[f'p{p}_most_prob_prob'] = qaoa_result['most_probable_prob']
                p_results[f'p{p}_iterations'] = qaoa_result['num_iterations']
                p_results[f'p{p}_time'] = p_time
                
                print(f"ratio={approx_ratio:.4f}, time={p_time:.1f}s")
            
            results_data.append(p_results)
            
            # Summary for this graph
            best_p = max(P_VALUES_TO_TEST, key=lambda p: p_results[f'p{p}_approx_ratio'])
            best_ratio = p_results[f'p{best_p}_approx_ratio']
            print(f"    ✓ Best performance: p={best_p} with ratio={best_ratio:.4f}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Store failed result
            p_results = {
                'N': n_qubits,
                'Graph_ID': graph_id,
                'Delta_min': delta_min,
                's_at_min': row['s_at_min'],
                'Max_degeneracy': row['Max_degeneracy'],
                'Optimal_cut': optimal_cut,
            }
            # Fill with -1 for all p values
            for p in P_VALUES_TO_TEST:
                p_results[f'p{p}_expected_cut'] = -1
                p_results[f'p{p}_approx_ratio'] = -1
                p_results[f'p{p}_most_prob_cut'] = -1
                p_results[f'p{p}_most_prob_prob'] = -1
                p_results[f'p{p}_iterations'] = -1
                p_results[f'p{p}_time'] = -1
            
            results_data.append(p_results)
    
    total_time = time.time() - total_start_time
    
    # Save results
    print("\n" + "-" * 70)
    print(f"\n💾 Saving results to {OUTPUT_FILENAME}...")
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_FILENAME, index=False)
    
    # Statistics for each p value
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   • Total graphs processed: {len(results_df)}")
    print(f"   • Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"   • Average time per graph: {total_time/len(results_df):.2f}s")
    
    print(f"\n📈 QAOA Performance Statistics by p:")
    
    for p in P_VALUES_TO_TEST:
        ratio_col = f'p{p}_approx_ratio'
        if ratio_col in results_df.columns:
            valid = results_df[results_df[ratio_col] >= 0]
            if len(valid) > 0:
                mean_ratio = valid[ratio_col].mean()
                std_ratio = valid[ratio_col].std()
                min_ratio = valid[ratio_col].min()
                max_ratio = valid[ratio_col].max()
                print(f"\n   p={p:2d}: Mean ratio = {mean_ratio:.4f} ± {std_ratio:.4f}")
                print(f"         Min = {min_ratio:.4f}, Max = {max_ratio:.4f}")
    
    # Find which p is typically needed for different thresholds
    print(f"\n🎯 p* Analysis (minimum p to reach threshold):")
    for threshold in [0.90, 0.95, 0.99]:
        p_star_values = []
        for idx, row in results_df.iterrows():
            for p in P_VALUES_TO_TEST:
                ratio_col = f'p{p}_approx_ratio'
                if row[ratio_col] >= threshold:
                    p_star_values.append(p)
                    break
            else:
                p_star_values.append(11)  # Target not reached
        
        p_star_series = pd.Series(p_star_values)
        reached = sum(p <= 10 for p in p_star_values)
        mean_p = p_star_series[p_star_series <= 10].mean() if reached > 0 else np.nan
        
        print(f"   Target {threshold:.2f}: {reached}/{len(results_df)} graphs reach it, mean p* = {mean_p:.2f}")
    
    # Correlation analysis
    print(f"\n🔬 Correlation Analysis (Δ_min vs approx_ratio):")
    for p in P_VALUES_TO_TEST:
        ratio_col = f'p{p}_approx_ratio'
        if ratio_col in results_df.columns:
            valid = results_df[(results_df[ratio_col] >= 0) & (results_df['Delta_min'] > 0)]
            if len(valid) > 1 and valid[ratio_col].std() > 0:
                corr = valid[['Delta_min', ratio_col]].corr().iloc[0, 1]
                print(f"   p={p:2d}: r = {corr:+.4f}")
    
    print(f"\n📁 Data saved to: {OUTPUT_FILENAME}")
    print("=" * 70)


# =========================================================================
# 6. COMMAND-LINE INTERFACE
# =========================================================================

def main():
    """Main execution function with mode selection."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Run 3-node validation example
        run_3node_example()
    else:
        # Run full analysis on graphs from CSV
        analyze_graphs_from_csv()


if __name__ == "__main__":
    main()

