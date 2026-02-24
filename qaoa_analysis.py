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
INPUT_CSV = 'outputs/spectral_gap/spectral_gap_3reg_N12_k2.csv'

# Degeneracy filtering (None = process all, int = filter to specific degeneracy)
FILTER_DEGENERACY = 4  # Only process deg=4 graphs (26 graphs from N12 dataset)

# QAOA parameters - p-sweep mode
P_VALUES_TO_TEST = list(range(1, 11))  # Test p=1,2,3,...,10
MAX_OPTIMIZER_ITERATIONS = 500  # Maximum classical optimizer iterations
OPTIMIZER_METHOD = 'COBYLA'     # Classical optimizer (COBYLA from tutorial)
NUM_SHOTS = 10000               # Number of measurement shots

# Output filename
OUTPUT_FILENAME = 'outputs/qaoa_unweighted/N12/qaoa_sweep_N12.csv'

# Optimization improvement parameters
USE_HEURISTIC_INIT = True          # Use problem-specific initialization for p=1
USE_WARMSTART = True               # Use warm-start from p-1 for p>=2
USE_MULTISTART = True              # Use multi-start for high p values
MULTISTART_THRESHOLD_P = 7         # Apply multi-start for p >= this value
MULTISTART_NUM_ATTEMPTS = 3        # Number of random starts to try
MULTISTART_EARLY_STOP_RATIO = 1 # Stop early if this ratio reached

# Simulation backend
SIMULATOR_METHOD = 'statevector'  # Noiseless simulation

# Random seed for reproducibility
RANDOM_SEED = 42

# Parallel processing configuration
USE_PARALLEL = True              # Enable parallel processing
NUM_WORKERS = None               # Number of workers (None = all CPU cores)
VERBOSE_PARALLEL = False         # Print detailed output in parallel mode

# =========================================================================
# 2. HELPER FUNCTIONS: BUILD COST HAMILTONIAN
# =========================================================================

def edges_to_cost_hamiltonian(edges: List[Tuple[int, int]], n_qubits: int,
                               weights: List[float] = None) -> SparsePauliOp:
    """
    Constructs the Max-Cut cost Hamiltonian as a SparsePauliOp.

    For Max-Cut: H_cost = ∑_{(i,j)∈E} w_{ij} Z_i Z_j

    Args:
        edges: List of edges as (vertex1, vertex2) tuples
        n_qubits: Number of qubits (nodes in graph)
        weights: Optional list of edge weights (default: all 1.0 for unweighted)

    Returns:
        SparsePauliOp representing the cost Hamiltonian
    """
    # Default to unit weights if not provided
    if weights is None:
        weights = [1.0] * len(edges)

    if len(weights) != len(edges):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of edges ({len(edges)})")

    # Build list of Pauli strings and coefficients
    pauli_list = []

    for (u, v), weight in zip(edges, weights):
        # Create Pauli string: Z on qubits u and v, I elsewhere
        pauli_str = ['I'] * n_qubits
        pauli_str[u] = 'Z'
        pauli_str[v] = 'Z'

        # Qiskit uses little-endian: reverse the string
        pauli_str_reversed = ''.join(pauli_str[::-1])

        # Add to list with edge weight as coefficient
        pauli_list.append((pauli_str_reversed, weight))

    # Create SparsePauliOp from list
    cost_hamiltonian = SparsePauliOp.from_list(pauli_list)

    return cost_hamiltonian


def evaluate_cut_value(bitstring: str, edges: List[Tuple[int, int]],
                       weights: List[float] = None) -> float:
    """
    Evaluate the cut value for a given bitstring assignment.

    Args:
        bitstring: Binary string (e.g., '01010')
        edges: List of edges
        weights: Optional list of edge weights (default: all 1.0 for unweighted)

    Returns:
        Cut value (sum of weights of cut edges, or count if unweighted)
    """
    # Default to unit weights if not provided
    if weights is None:
        weights = [1.0] * len(edges)

    cut_value = 0.0
    for (u, v), weight in zip(edges, weights):
        # Edge is in cut if endpoints have different values
        if bitstring[u] != bitstring[v]:
            cut_value += weight
    return cut_value


def calculate_expected_cut(counts: Dict[str, int], edges: List[Tuple[int, int]],
                           weights: List[float] = None) -> float:
    """
    Calculate expected cut value from measurement counts.

    Expected cut = ∑_bitstrings P(bitstring) × cut_value(bitstring)

    Args:
        counts: Dictionary mapping bitstrings to counts
        edges: List of edges
        weights: Optional list of edge weights (default: all 1.0 for unweighted)

    Returns:
        Expected cut value (as float)
    """
    expected_cut = 0.0
    total_shots = sum(counts.values())

    for bitstring, count in counts.items():
        prob = count / total_shots
        cut_val = evaluate_cut_value(bitstring, edges, weights)
        expected_cut += prob * cut_val

    return expected_cut


def calculate_success_probability(counts: Dict[str, int], edges: List[Tuple[int, int]],
                                   optimal_cut: float, weights: List[float] = None,
                                   tol: float = 1e-6) -> float:
    """
    Calculate the probability of measuring an optimal solution.

    P_success = sum of P(x) for all bitstrings x where cut_value(x) == optimal_cut.
    This naturally handles degeneracy — both bit-flip symmetric solutions
    (and any other degenerate optima) are counted.

    Args:
        counts: Dictionary mapping bitstrings to measurement counts
        edges: List of edges
        optimal_cut: Known optimal cut value
        weights: Optional list of edge weights (default: all 1.0 for unweighted)
        tol: Tolerance for floating-point comparison (for weighted graphs)

    Returns:
        Success probability in [0, 1]
    """
    total_shots = sum(counts.values())
    success_count = 0

    for bitstring, count in counts.items():
        cut_val = evaluate_cut_value(bitstring, edges, weights)
        if abs(cut_val - optimal_cut) < tol:
            success_count += count

    return success_count / total_shots


# =========================================================================
# 3. OPTIMIZATION IMPROVEMENT FUNCTIONS
# =========================================================================

def get_heuristic_initial_params(p: int) -> np.ndarray:
    """
    Generate heuristic initial parameters for MaxCut QAOA.
    Based on typical good ranges from literature:
    - γ ∈ [0, π/4] (interaction strength)
    - β ∈ [0, π/2] (mixing angle)

    Args:
        p: Number of QAOA layers

    Returns:
        Initial parameter array [γ₁, ..., γₚ, β₁, ..., βₚ]
    """
    gamma = np.linspace(0.1, np.pi/4, p)
    beta = np.linspace(0.1, np.pi/2, p)
    return np.concatenate([gamma, beta])


def extend_params_for_warmstart(previous_params: np.ndarray, p: int) -> np.ndarray:
    """
    Extend previous optimal parameters for next layer (warm-start).
    Strategy: Add new layer with slight perturbation of previous layer's last values.

    Args:
        previous_params: Optimal parameters from p-1 run
        p: Target number of layers (should be len(previous_params)//2 + 1)

    Returns:
        Extended parameter array for p layers
    """
    p_prev = len(previous_params) // 2
    gamma_prev = previous_params[:p_prev]
    beta_prev = previous_params[p_prev:]

    # New layer parameters with slight decay and small random perturbation
    np.random.seed(RANDOM_SEED + p)  # Deterministic but p-dependent
    gamma_new = gamma_prev[-1] * 0.9 + 0.05 * np.random.randn()
    beta_new = beta_prev[-1] * 0.9 + 0.05 * np.random.randn()

    # Clip to valid ranges
    gamma_new = np.clip(gamma_new, 0, np.pi)
    beta_new = np.clip(beta_new, 0, np.pi)

    return np.concatenate([gamma_prev, [gamma_new], beta_prev, [beta_new]])


def run_qaoa_multistart(edges: List[Tuple[int, int]],
                        n_qubits: int,
                        p: int,
                        max_iter: int,
                        initial_params: np.ndarray = None,
                        optimal_cut: float = None,
                        num_attempts: int = 3,
                        early_stop_ratio: float = 0.95,
                        weights: List[float] = None) -> Dict:
    """
    Run QAOA with multiple random initializations, keeping the best result.

    Args:
        edges: Graph edges
        n_qubits: Number of qubits
        p: Number of QAOA layers
        max_iter: Max optimizer iterations per attempt
        initial_params: First attempt uses this (e.g., warm-start), others are random
        optimal_cut: Known optimal cut value (for early stopping)
        num_attempts: Number of random starts to try
        early_stop_ratio: Stop if approximation ratio reaches this threshold
        weights: Optional list of edge weights (default: all 1.0 for unweighted)

    Returns:
        Best result dictionary from all attempts
    """
    best_result = None
    best_expected_cut = -np.inf

    for attempt in range(num_attempts):
        # First attempt uses provided initial_params (e.g., warm-start)
        # Subsequent attempts use random initialization
        if attempt == 0 and initial_params is not None:
            attempt_params = initial_params
        else:
            np.random.seed(RANDOM_SEED + attempt + p * 100)
            attempt_params = 2 * np.pi * np.random.rand(2 * p)

        print(f"      Multi-start attempt {attempt+1}/{num_attempts}...", end=" ")

        result = run_qaoa_single(
            edges=edges,
            n_qubits=n_qubits,
            p=p,
            max_iter=max_iter,
            initial_params=attempt_params,
            weights=weights,
            optimal_cut=optimal_cut
        )

        expected_cut = result['expected_cut']
        print(f"cut={expected_cut:.2f}")

        if expected_cut > best_expected_cut:
            best_expected_cut = expected_cut
            best_result = result
            best_result['multistart_attempts_used'] = attempt + 1

        # Early stopping if target ratio reached
        if optimal_cut is not None:
            ratio = expected_cut / optimal_cut
            if ratio >= early_stop_ratio:
                print(f"      Early stop: ratio {ratio:.4f} >= {early_stop_ratio:.4f}")
                break

    return best_result


# =========================================================================
# 4. QAOA OPTIMIZATION FUNCTION
# =========================================================================

def run_qaoa_single(edges: List[Tuple[int, int]],
                    n_qubits: int,
                    p: int = 1,
                    max_iter: int = 200,
                    initial_params: np.ndarray = None,
                    weights: List[float] = None,
                    optimal_cut: float = None) -> Dict:
    """
    Run QAOA on a Max-Cut problem instance (single optimization attempt).

    Args:
        edges: Graph edges as list of tuples
        n_qubits: Number of qubits (nodes)
        p: Number of QAOA layers
        max_iter: Maximum optimizer iterations
        initial_params: Initial parameter values (random if None)
        weights: Optional list of edge weights (default: all 1.0 for unweighted)

    Returns:
        Dictionary with results:
        - 'best_bitstring': Best solution found
        - 'best_cut_value': Cut value of best solution
        - 'num_iterations': Number of optimizer iterations
        - 'final_cost': Final cost function value
        - 'optimization_time': Time spent in optimization (seconds)
        - 'optimal_params': Optimal parameters found
    """
    # Default to unit weights if not provided
    if weights is None:
        weights = [1.0] * len(edges)

    # Build cost Hamiltonian
    cost_hamiltonian = edges_to_cost_hamiltonian(edges, n_qubits, weights)

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

            # Calculate energy for this bitstring (weighted)
            energy = 0.0
            for (u, v), weight in zip(edges, weights):
                # Z_i Z_j eigenvalue: +w if same, -w if different
                if bitstring_reversed[u] == bitstring_reversed[v]:
                    energy += weight
                else:
                    energy -= weight

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
    most_probable_cut = evaluate_cut_value(most_probable_bitstring, edges, weights)

    # Calculate expected cut value (correct approximation ratio)
    expected_cut = calculate_expected_cut(counts_reversed, edges, weights)

    # Calculate success probability if optimal_cut is known
    success_prob = None
    if optimal_cut is not None:
        success_prob = calculate_success_probability(
            counts_reversed, edges, optimal_cut, weights)

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
        'success_prob': success_prob,
        'num_iterations': iteration_count[0],
        'final_cost': result.fun,
        'optimization_time': optimization_time,
        'optimal_params': optimal_params,
        'cost_history': cost_history
    }


def run_qaoa(edges: List[Tuple[int, int]],
             n_qubits: int,
             p: int = 1,
             max_iter: int = 200,
             initial_params: np.ndarray = None,
             optimal_cut: float = None,
             weights: List[float] = None) -> Dict:
    """
    Run QAOA with automatic multi-start for high p values.

    Dispatcher that chooses between single-start and multi-start optimization
    based on configuration and p value.

    Args:
        edges: Graph edges as list of tuples
        n_qubits: Number of qubits (nodes)
        p: Number of QAOA layers
        max_iter: Maximum optimizer iterations
        initial_params: Initial parameter values (for warm-start or heuristic)
        optimal_cut: Known optimal cut value (for early stopping in multi-start)
        weights: Optional list of edge weights (default: all 1.0 for unweighted)

    Returns:
        Dictionary with QAOA results (same format as run_qaoa_single)
    """
    # Use multi-start for high p values if enabled
    if USE_MULTISTART and p >= MULTISTART_THRESHOLD_P:
        return run_qaoa_multistart(
            edges=edges,
            n_qubits=n_qubits,
            p=p,
            max_iter=max_iter,
            initial_params=initial_params,
            optimal_cut=optimal_cut,
            num_attempts=MULTISTART_NUM_ATTEMPTS,
            early_stop_ratio=MULTISTART_EARLY_STOP_RATIO,
            weights=weights
        )
    else:
        return run_qaoa_single(
            edges=edges,
            n_qubits=n_qubits,
            p=p,
            max_iter=max_iter,
            initial_params=initial_params,
            weights=weights,
            optimal_cut=optimal_cut
        )


# =========================================================================
# 5. PARALLEL PROCESSING SUPPORT
# =========================================================================

def process_single_graph(args):
    """
    Process a single graph - worker function for multiprocessing.

    This function is designed to be called by multiprocessing.Pool.map().
    Each worker gets its own copy of the simulator and configuration.

    Args:
        args: Tuple of (graph_data, config) where:
            - graph_data: dict with graph_id, n_qubits, delta_min, s_at_min,
                         max_degeneracy, optimal_cut, edges
            - config: dict with p_values, max_iter, num_shots, simulator_method,
                     random_seed, use_heuristic_init, use_warmstart, use_multistart,
                     multistart_threshold_p, multistart_num_attempts,
                     multistart_early_stop_ratio, verbose

    Returns:
        Dictionary with results for all p values
    """
    graph_data, config = args

    # Unpack graph data
    graph_id = graph_data['graph_id']
    n_qubits = graph_data['n_qubits']
    delta_min = graph_data['delta_min']
    s_at_min = graph_data['s_at_min']
    max_degeneracy = graph_data['max_degeneracy']
    optimal_cut = graph_data['optimal_cut']
    edges = graph_data['edges']
    idx = graph_data['idx']
    total = graph_data['total']

    # Unpack config
    p_values = config['p_values']
    max_iter = config['max_iter']
    verbose = config.get('verbose', False)

    # Set module-level configuration for this worker
    global NUM_SHOTS, SIMULATOR_METHOD, RANDOM_SEED
    global USE_HEURISTIC_INIT, USE_WARMSTART, USE_MULTISTART
    global MULTISTART_THRESHOLD_P, MULTISTART_NUM_ATTEMPTS, MULTISTART_EARLY_STOP_RATIO
    global MAX_OPTIMIZER_ITERATIONS

    NUM_SHOTS = config['num_shots']
    SIMULATOR_METHOD = config['simulator_method']
    RANDOM_SEED = config['random_seed'] + graph_id  # Unique seed per graph
    USE_HEURISTIC_INIT = config['use_heuristic_init']
    USE_WARMSTART = config['use_warmstart']
    USE_MULTISTART = config['use_multistart']
    MULTISTART_THRESHOLD_P = config['multistart_threshold_p']
    MULTISTART_NUM_ATTEMPTS = config['multistart_num_attempts']
    MULTISTART_EARLY_STOP_RATIO = config['multistart_early_stop_ratio']
    MAX_OPTIMIZER_ITERATIONS = max_iter

    # Start timing
    graph_start = time.time()

    if verbose:
        print(f"[{idx+1}/{total}] Graph #{graph_id} | N={n_qubits} | Δ_min={delta_min:.6f}")

    # Initialize results
    p_results = {
        'N': n_qubits,
        'Graph_ID': graph_id,
        'Delta_min': delta_min,
        's_at_min': s_at_min,
        'Max_degeneracy': max_degeneracy,
        'Optimal_cut': optimal_cut,
    }

    try:
        # Track previous optimal params for warm-start
        previous_optimal_params = None

        # Test each p value
        for p in p_values:
            p_start = time.time()

            # Determine initial parameters
            if USE_WARMSTART and previous_optimal_params is not None:
                initial_params = extend_params_for_warmstart(previous_optimal_params, p)
            elif USE_HEURISTIC_INIT and p == 1:
                initial_params = get_heuristic_initial_params(p)
            else:
                initial_params = None

            # Run QAOA (use run_qaoa_single directly to avoid verbose output)
            if USE_MULTISTART and p >= MULTISTART_THRESHOLD_P:
                qaoa_result = run_qaoa_multistart_quiet(
                    edges=edges,
                    n_qubits=n_qubits,
                    p=p,
                    max_iter=max_iter,
                    initial_params=initial_params,
                    optimal_cut=optimal_cut,
                    num_attempts=MULTISTART_NUM_ATTEMPTS,
                    early_stop_ratio=MULTISTART_EARLY_STOP_RATIO
                )
            else:
                qaoa_result = run_qaoa_single_quiet(
                    edges=edges,
                    n_qubits=n_qubits,
                    p=p,
                    max_iter=max_iter,
                    initial_params=initial_params,
                    optimal_cut=optimal_cut
                )

            # Store optimal params for next iteration
            previous_optimal_params = qaoa_result['optimal_params']

            p_time = time.time() - p_start

            # Calculate approximation ratio
            expected_cut = qaoa_result['expected_cut']
            approx_ratio = expected_cut / optimal_cut

            # Store results
            p_results[f'p{p}_expected_cut'] = expected_cut
            p_results[f'p{p}_approx_ratio'] = approx_ratio
            p_results[f'p{p}_success_prob'] = qaoa_result.get('success_prob', -1)
            p_results[f'p{p}_most_prob_cut'] = qaoa_result['most_probable_cut']
            p_results[f'p{p}_most_prob_prob'] = qaoa_result['most_probable_prob']
            p_results[f'p{p}_iterations'] = qaoa_result['num_iterations']
            p_results[f'p{p}_time'] = p_time

        graph_time = time.time() - graph_start
        best_p = max(p_values, key=lambda p: p_results[f'p{p}_approx_ratio'])
        best_ratio = p_results[f'p{best_p}_approx_ratio']

        print(f"✓ Graph #{graph_id:3d} done: best p={best_p}, ratio={best_ratio:.4f}, time={graph_time:.1f}s")

    except Exception as e:
        print(f"✗ Graph #{graph_id} failed: {e}")
        for p in p_values:
            p_results[f'p{p}_expected_cut'] = -1
            p_results[f'p{p}_approx_ratio'] = -1
            p_results[f'p{p}_success_prob'] = -1
            p_results[f'p{p}_most_prob_cut'] = -1
            p_results[f'p{p}_most_prob_prob'] = -1
            p_results[f'p{p}_iterations'] = -1
            p_results[f'p{p}_time'] = -1

    return p_results


def run_qaoa_single_quiet(edges, n_qubits, p=1, max_iter=200, initial_params=None, weights=None, optimal_cut=None):
    """
    Run QAOA without verbose output - for parallel processing.
    Same as run_qaoa_single but without print statements.
    """
    # Default to unit weights if not provided
    if weights is None:
        weights = [1.0] * len(edges)

    # Build cost Hamiltonian
    cost_hamiltonian = edges_to_cost_hamiltonian(edges, n_qubits, weights)

    # Create QAOA ansatz circuit
    qaoa_circuit = QAOAAnsatz(cost_hamiltonian, reps=p)

    # Setup simulator
    backend = AerSimulator(method=SIMULATOR_METHOD)

    # Transpile circuit for backend
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled_circuit = pm.run(qaoa_circuit)

    # Counter for iterations
    iteration_count = [0]

    # Cost function to minimize (quiet version)
    def cost_function(params):
        iteration_count[0] += 1
        bound_circuit = transpiled_circuit.assign_parameters(params)
        from qiskit import QuantumCircuit
        measured_circuit = bound_circuit.copy()
        measured_circuit.measure_all()

        job = backend.run(measured_circuit, shots=NUM_SHOTS, seed_simulator=RANDOM_SEED)
        result = job.result()
        counts = result.get_counts()

        expectation = 0.0
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            bitstring_reversed = bitstring[::-1]
            energy = 0.0
            for (u, v), weight in zip(edges, weights):
                if bitstring_reversed[u] == bitstring_reversed[v]:
                    energy += weight
                else:
                    energy -= weight
            expectation += (count / total_shots) * energy

        return expectation

    # Initial parameters
    if initial_params is None:
        np.random.seed(RANDOM_SEED)
        initial_params = 2 * np.pi * np.random.rand(2 * p)

    # Run optimization
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

    # Sample from final circuit
    from qiskit import QuantumCircuit
    bound_circuit_final = transpiled_circuit.assign_parameters(optimal_params)
    measured_circuit_final = bound_circuit_final.copy()
    measured_circuit_final.measure_all()

    job_final = backend.run(measured_circuit_final, shots=NUM_SHOTS, seed_simulator=RANDOM_SEED)
    result_final = job_final.result()
    counts_final = result_final.get_counts()

    counts_reversed = {}
    for bitstring, count in counts_final.items():
        counts_reversed[bitstring[::-1]] = count

    most_probable_bitstring = max(counts_reversed, key=counts_reversed.get)
    most_probable_count = counts_reversed[most_probable_bitstring]
    most_probable_prob = most_probable_count / NUM_SHOTS
    most_probable_cut = evaluate_cut_value(most_probable_bitstring, edges, weights)
    expected_cut = calculate_expected_cut(counts_reversed, edges, weights)

    success_prob = None
    if optimal_cut is not None:
        success_prob = calculate_success_probability(
            counts_reversed, edges, optimal_cut, weights)

    return {
        'best_bitstring': most_probable_bitstring,
        'best_cut_value': most_probable_cut,
        'most_probable_bitstring': most_probable_bitstring,
        'most_probable_prob': most_probable_prob,
        'most_probable_cut': most_probable_cut,
        'expected_cut': expected_cut,
        'success_prob': success_prob,
        'num_iterations': iteration_count[0],
        'final_cost': result.fun,
        'optimization_time': optimization_time,
        'optimal_params': optimal_params,
    }


def run_qaoa_multistart_quiet(edges, n_qubits, p, max_iter, initial_params=None,
                               optimal_cut=None, num_attempts=3, early_stop_ratio=0.95,
                               weights=None):
    """
    Run QAOA multistart without verbose output - for parallel processing.
    """
    best_result = None
    best_expected_cut = -np.inf

    for attempt in range(num_attempts):
        if attempt == 0 and initial_params is not None:
            attempt_params = initial_params
        else:
            np.random.seed(RANDOM_SEED + attempt + p * 100)
            attempt_params = 2 * np.pi * np.random.rand(2 * p)

        result = run_qaoa_single_quiet(
            edges=edges,
            n_qubits=n_qubits,
            p=p,
            max_iter=max_iter,
            initial_params=attempt_params,
            weights=weights,
            optimal_cut=optimal_cut
        )

        expected_cut = result['expected_cut']

        if expected_cut > best_expected_cut:
            best_expected_cut = expected_cut
            best_result = result
            best_result['multistart_attempts_used'] = attempt + 1

        if optimal_cut is not None:
            ratio = expected_cut / optimal_cut
            if ratio >= early_stop_ratio:
                break

    return best_result


def analyze_graphs_parallel(df, p_values, output_filename, num_workers=None, config=None):
    """
    Parallel version of graph analysis using multiprocessing.Pool.

    Args:
        df: DataFrame with graph data
        p_values: List of p values to test
        output_filename: Path to save results
        num_workers: Number of worker processes (None = all CPUs)
        config: Configuration dictionary

    Returns:
        DataFrame with results
    """
    from multiprocessing import Pool, cpu_count

    num_workers = num_workers or cpu_count()

    print(f"\n🚀 Starting PARALLEL QAOA analysis")
    print(f"   • Workers: {num_workers}")
    print(f"   • Graphs: {len(df)}")
    print(f"   • P values: {p_values}")
    print("-" * 70)

    # Prepare work items
    work_items = []
    for idx, row in df.iterrows():
        graph_data = {
            'graph_id': row['Graph_ID'],
            'n_qubits': row['N'],
            'delta_min': row['Delta_min'],
            's_at_min': row['s_at_min'],
            'max_degeneracy': row['Max_degeneracy'],
            'optimal_cut': row['Max_cut_value'],
            'edges': ast.literal_eval(row['Edges']) if isinstance(row['Edges'], str) else row['Edges'],
            'idx': idx,
            'total': len(df),
        }
        work_items.append((graph_data, config))

    # Run parallel processing
    total_start_time = time.time()

    with Pool(num_workers) as pool:
        results_data = pool.map(process_single_graph, work_items)

    total_time = time.time() - total_start_time

    # Save results
    print("\n" + "-" * 70)
    print(f"\n💾 Saving results to {output_filename}...")
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_filename, index=False)

    # Print statistics
    print(f"\n✅ PARALLEL ANALYSIS COMPLETE!")
    print(f"   • Total graphs processed: {len(results_df)}")
    print(f"   • Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"   • Average time per graph: {total_time/len(results_df):.2f}s")
    print(f"   • Speedup factor: ~{num_workers}x (using {num_workers} workers)")

    # Performance stats
    print(f"\n📈 QAOA Performance Statistics by p:")
    for p in p_values:
        ratio_col = f'p{p}_approx_ratio'
        if ratio_col in results_df.columns:
            valid = results_df[results_df[ratio_col] >= 0]
            if len(valid) > 0:
                mean_ratio = valid[ratio_col].mean()
                std_ratio = valid[ratio_col].std()
                print(f"   p={p:2d}: mean={mean_ratio:.4f}±{std_ratio:.4f}")

    print(f"\n📁 Data saved to: {output_filename}")
    print("=" * 70)

    return results_df


# =========================================================================
# 6. EXAMPLE: 3-NODE TRIANGLE (VALIDATION)
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
    test_p = 2  # Test with p=2 layers

    print(f"\nGraph: {edges}")
    print(f"Optimal cut value: {optimal_cut}")
    print(f"Configuration: p={test_p}, max_iter={MAX_OPTIMIZER_ITERATIONS}, shots={NUM_SHOTS}")

    # Run QAOA
    result = run_qaoa(edges, n_qubits, p=test_p, max_iter=MAX_OPTIMIZER_ITERATIONS, optimal_cut=optimal_cut)

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
# 6. MAIN: ANALYZE GRAPHS FROM SPECTRAL GAP DATA
# =========================================================================

def analyze_graphs_from_csv(csv_path: str = INPUT_CSV, use_parallel: bool = None,
                            num_workers: int = None):
    """
    Load graphs from spectral gap analysis CSV and run QAOA p-sweep on each.

    Tests multiple p values (1 through 10) for each graph and saves all results.
    Supports both sequential and parallel processing modes.

    Args:
        csv_path: Path to CSV file with spectral gap data
        use_parallel: Override USE_PARALLEL config (None = use config)
        num_workers: Override NUM_WORKERS config (None = use config or all CPUs)
    """
    from multiprocessing import cpu_count

    # Determine parallel mode
    parallel_mode = use_parallel if use_parallel is not None else USE_PARALLEL
    workers = num_workers if num_workers is not None else (NUM_WORKERS or cpu_count())

    print("=" * 70)
    print("  QAOA P-SWEEP ANALYSIS ON SPECTRAL GAP GRAPHS")
    print("=" * 70)

    # Load data
    print(f"\n📖 Loading graph data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} graphs for N={df['N'].iloc[0]}")

    # Filter by degeneracy if specified
    if FILTER_DEGENERACY is not None:
        df = df[df['Max_degeneracy'] == FILTER_DEGENERACY].copy()
        df = df.reset_index(drop=True)
        print(f"   Filtered to {len(df)} graphs with degeneracy={FILTER_DEGENERACY}")

    print(f"\n📊 Configuration:")
    print(f"   • QAOA layers to test: p={P_VALUES_TO_TEST}")
    print(f"   • Max optimizer iterations: {MAX_OPTIMIZER_ITERATIONS}")
    print(f"   • Optimizer method: {OPTIMIZER_METHOD}")
    print(f"   • Number of shots: {NUM_SHOTS}")
    print(f"   • Simulator method: {SIMULATOR_METHOD}")
    print(f"   • Random seed: {RANDOM_SEED}")
    print(f"\n📈 Optimization Improvements:")
    print(f"   • Heuristic initialization (p=1): {'Enabled' if USE_HEURISTIC_INIT else 'Disabled'}")
    print(f"   • Warm-start (p≥2): {'Enabled' if USE_WARMSTART else 'Disabled'}")
    print(f"   • Multi-start (p≥{MULTISTART_THRESHOLD_P}): {'Enabled' if USE_MULTISTART else 'Disabled'}")
    if USE_MULTISTART:
        print(f"     - Attempts: {MULTISTART_NUM_ATTEMPTS}")
        print(f"     - Early stop ratio: {MULTISTART_EARLY_STOP_RATIO}")

    print(f"\n⚡ Parallel Processing: {'Enabled (' + str(workers) + ' workers)' if parallel_mode else 'Disabled'}")

    # Use parallel processing if enabled
    if parallel_mode:
        config = {
            'p_values': P_VALUES_TO_TEST,
            'max_iter': MAX_OPTIMIZER_ITERATIONS,
            'num_shots': NUM_SHOTS,
            'simulator_method': SIMULATOR_METHOD,
            'random_seed': RANDOM_SEED,
            'use_heuristic_init': USE_HEURISTIC_INIT,
            'use_warmstart': USE_WARMSTART,
            'use_multistart': USE_MULTISTART,
            'multistart_threshold_p': MULTISTART_THRESHOLD_P,
            'multistart_num_attempts': MULTISTART_NUM_ATTEMPTS,
            'multistart_early_stop_ratio': MULTISTART_EARLY_STOP_RATIO,
            'verbose': VERBOSE_PARALLEL,
        }
        return analyze_graphs_parallel(df, P_VALUES_TO_TEST, OUTPUT_FILENAME,
                                       num_workers=workers, config=config)

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

            # Track previous optimal params for warm-start
            previous_optimal_params = None

            # Test each p value
            for p in P_VALUES_TO_TEST:
                print(f"    Testing p={p}...", end=" ")
                p_start = time.time()

                # Determine initial parameters using optimization strategy
                if USE_WARMSTART and previous_optimal_params is not None:
                    initial_params = extend_params_for_warmstart(previous_optimal_params, p)
                elif USE_HEURISTIC_INIT and p == 1:
                    initial_params = get_heuristic_initial_params(p)
                else:
                    initial_params = None  # Random initialization

                qaoa_result = run_qaoa(
                    edges=edges,
                    n_qubits=n_qubits,
                    p=p,
                    max_iter=MAX_OPTIMIZER_ITERATIONS,
                    initial_params=initial_params,
                    optimal_cut=optimal_cut
                )

                # Store optimal params for next iteration
                previous_optimal_params = qaoa_result['optimal_params']

                p_time = time.time() - p_start

                # Calculate approximation ratio (using expected cut)
                expected_cut = qaoa_result['expected_cut']
                approx_ratio = expected_cut / optimal_cut

                # Store results for this p
                p_results[f'p{p}_expected_cut'] = expected_cut
                p_results[f'p{p}_approx_ratio'] = approx_ratio
                p_results[f'p{p}_success_prob'] = qaoa_result.get('success_prob', -1)
                p_results[f'p{p}_most_prob_cut'] = qaoa_result['most_probable_cut']
                p_results[f'p{p}_most_prob_prob'] = qaoa_result['most_probable_prob']
                p_results[f'p{p}_iterations'] = qaoa_result['num_iterations']
                p_results[f'p{p}_time'] = p_time

                print(f"ratio={approx_ratio:.4f}, success_prob={qaoa_result.get('success_prob', 'N/A')}, time={p_time:.1f}s")

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
                p_results[f'p{p}_success_prob'] = -1
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
# 7. COMMAND-LINE INTERFACE
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

