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
INPUT_CSV = 'outputs/Delta_min_3_regular_N10_res200.csv'

# QAOA parameters
P_LAYERS = 1                    # Number of QAOA layers (p)
MAX_OPTIMIZER_ITERATIONS = 200  # Maximum classical optimizer iterations
OPTIMIZER_METHOD = 'COBYLA'     # Classical optimizer (COBYLA from tutorial)
NUM_SHOTS = 10000               # Number of measurement shots

# Output filename
OUTPUT_FILENAME = 'outputs/QAOA_results_N10_p1.csv'

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
    
    # Find bitstring with highest count
    best_bitstring = max(counts_final, key=counts_final.get)
    best_bitstring_reversed = best_bitstring[::-1]  # Qiskit little-endian
    
    # Calculate cut value for best solution
    best_cut_value = evaluate_cut_value(best_bitstring_reversed, edges)
    
    return {
        'best_bitstring': best_bitstring_reversed,
        'best_cut_value': best_cut_value,
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
    Load graphs from spectral gap analysis CSV and run QAOA on each.
    
    Args:
        csv_path: Path to CSV file with spectral gap data
    """
    print("=" * 70)
    print("  QAOA ANALYSIS ON SPECTRAL GAP GRAPHS")
    print("=" * 70)
    
    # Load data
    print(f"\n📖 Loading graph data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} graphs for N={df['N'].iloc[0]}")
    
    print(f"\n📊 Configuration:")
    print(f"   • QAOA layers (p): {P_LAYERS}")
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
        
        # Run QAOA
        try:
            qaoa_result = run_qaoa(
                edges=edges,
                n_qubits=n_qubits,
                p=P_LAYERS,
                max_iter=MAX_OPTIMIZER_ITERATIONS
            )
            
            # Calculate approximation ratio
            approx_ratio = qaoa_result['best_cut_value'] / optimal_cut
            
            # Store results
            results_data.append({
                'N': n_qubits,
                'Graph_ID': graph_id,
                'Delta_min': delta_min,
                's_at_min': row['s_at_min'],
                'Max_degeneracy': row['Max_degeneracy'],
                'Optimal_cut': optimal_cut,
                'p_layers': P_LAYERS,
                'QAOA_cut_value': qaoa_result['best_cut_value'],
                'Approximation_ratio': approx_ratio,
                'Optimizer_iterations': qaoa_result['num_iterations'],
                'Final_cost': qaoa_result['final_cost'],
                'Optimization_time': qaoa_result['optimization_time'],
                'Best_bitstring': qaoa_result['best_bitstring']
            })
            
            print(f"    ✓ Result: Cut={qaoa_result['best_cut_value']}/{optimal_cut}, "
                  f"Ratio={approx_ratio:.4f}, Iters={qaoa_result['num_iterations']}, "
                  f"Time={qaoa_result['optimization_time']:.1f}s")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            # Store failed result
            results_data.append({
                'N': n_qubits,
                'Graph_ID': graph_id,
                'Delta_min': delta_min,
                's_at_min': row['s_at_min'],
                'Max_degeneracy': row['Max_degeneracy'],
                'Optimal_cut': optimal_cut,
                'p_layers': P_LAYERS,
                'QAOA_cut_value': -1,
                'Approximation_ratio': -1,
                'Optimizer_iterations': -1,
                'Final_cost': np.nan,
                'Optimization_time': -1,
                'Best_bitstring': 'ERROR'
            })
    
    total_time = time.time() - total_start_time
    
    # Save results
    print("\n" + "-" * 70)
    print(f"\n💾 Saving results to {OUTPUT_FILENAME}...")
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_FILENAME, index=False)
    
    # Statistics
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   • Total graphs processed: {len(results_df)}")
    print(f"   • Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"   • Average time per graph: {total_time/len(results_df):.2f}s")
    
    # Filter successful results
    success_df = results_df[results_df['Approximation_ratio'] >= 0]
    
    if len(success_df) > 0:
        print(f"\n📈 QAOA Performance Statistics:")
        print(f"   • Successful runs: {len(success_df)}/{len(results_df)}")
        print(f"   • Mean approximation ratio: {success_df['Approximation_ratio'].mean():.6f}")
        print(f"   • Std approximation ratio: {success_df['Approximation_ratio'].std():.6f}")
        print(f"   • Min approximation ratio: {success_df['Approximation_ratio'].min():.6f}")
        print(f"   • Max approximation ratio: {success_df['Approximation_ratio'].max():.6f}")
        print(f"   • Mean optimizer iterations: {success_df['Optimizer_iterations'].mean():.1f}")
        
        # Correlation hint
        print(f"\n🔬 Correlation Analysis Preview:")
        correlation = success_df[['Delta_min', 'Approximation_ratio']].corr().iloc[0, 1]
        print(f"   • Correlation(Δ_min, Approx_ratio): {correlation:.6f}")
        
        # Find hardest graph
        hardest_idx = success_df['Approximation_ratio'].idxmin()
        hardest = success_df.loc[hardest_idx]
        print(f"\n   Hardest graph for QAOA:")
        print(f"   • Graph ID: {hardest['Graph_ID']}")
        print(f"   • Approximation ratio: {hardest['Approximation_ratio']:.6f}")
        print(f"   • Δ_min: {hardest['Delta_min']:.6f}")
        
        # Find easiest graph
        easiest_idx = success_df['Approximation_ratio'].idxmax()
        easiest = success_df.loc[easiest_idx]
        print(f"\n   Easiest graph for QAOA:")
        print(f"   • Graph ID: {easiest['Graph_ID']}")
        print(f"   • Approximation ratio: {easiest['Approximation_ratio']:.6f}")
        print(f"   • Δ_min: {easiest['Delta_min']:.6f}")
    
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

