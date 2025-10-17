#!/usr/bin/env python3
"""
=========================================================================
QAOA Single Graph Visualizer
=========================================================================
Interactive visualization tool to see QAOA behavior on individual graphs.

Features:
1. Graph structure visualization (before/after)
2. Probability distribution over measurement outcomes
3. Cut value histogram
4. Performance metrics

Usage:
    python3 visualize_qaoa_single_graph.py                    # Default: Graph #2
    python3 visualize_qaoa_single_graph.py --graph_id 5       # Specific graph
    python3 visualize_qaoa_single_graph.py --hardest          # Smallest Δ_min
    python3 visualize_qaoa_single_graph.py --easiest          # Largest Δ_min
    python3 visualize_qaoa_single_graph.py --example triangle # 3-node example
=========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import List, Tuple, Dict
import ast
import time

# Import from qaoa_analysis to reuse code
from qaoa_analysis import (
    edges_to_cost_hamiltonian,
    evaluate_cut_value,
    calculate_expected_cut,
)

# Qiskit imports
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize

# =========================================================================
# CONFIGURATION
# =========================================================================

# Data source
INPUT_CSV = 'outputs/Delta_min_3_regular_N12_res20.csv'
DEFAULT_GRAPH_ID = 5  # Smallest Δ_min (hardest)

# QAOA parameters
P_LAYERS = 1
MAX_OPTIMIZER_ITERATIONS = 200
NUM_SHOTS = 10000
OPTIMIZER_METHOD = 'COBYLA'
SIMULATOR_METHOD = 'statevector'
RANDOM_SEED = 42

# Visualization parameters
NUM_TOP_BITSTRINGS = 25        # Number of bars to show in distribution
GRAPH_LAYOUT = 'circular'      # 'circular', 'spring', 'shell', 'kamada_kawai'
SAVE_FIGURE = True
OUTPUT_DIR = 'outputs/'
FIGSIZE = (20, 10)
DPI = 300

# =========================================================================
# PREDEFINED EXAMPLES
# =========================================================================

EXAMPLES = {
    'triangle': {
        'edges': [(0, 1), (1, 2), (0, 2)],
        'n_qubits': 3,
        'optimal_cut': 2
    },
    'square': {
        'edges': [(0, 1), (1, 2), (2, 3), (3, 0)],
        'n_qubits': 4,
        'optimal_cut': 4
    },
    'pentagon': {
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        'n_qubits': 5,
        'optimal_cut': 4
    }
}

# =========================================================================
# GRAPH LOADING
# =========================================================================

def load_graph_from_csv(graph_id: int, csv_path: str = INPUT_CSV) -> Dict:
    """
    Load specific graph and its properties from spectral gap data.
    
    Returns:
        Dictionary with graph properties
    """
    df = pd.read_csv(csv_path)
    row = df[df['Graph_ID'] == graph_id]
    
    if len(row) == 0:
        raise ValueError(f"Graph ID {graph_id} not found in {csv_path}")
    
    row = row.iloc[0]
    edges = ast.literal_eval(row['Edges'])
    
    return {
        'graph_id': int(row['Graph_ID']),
        'n_qubits': int(row['N']),
        'edges': edges,
        'delta_min': float(row['Delta_min']),
        's_at_min': float(row['s_at_min']),
        'max_degeneracy': int(row['Max_degeneracy']),
        'optimal_cut': int(row['Max_cut_value'])
    }


def get_hardest_graph(csv_path: str = INPUT_CSV) -> Dict:
    """Load graph with smallest Δ_min (hardest)."""
    df = pd.read_csv(csv_path)
    hardest_idx = df['Delta_min'].idxmin()
    return load_graph_from_csv(int(df.loc[hardest_idx, 'Graph_ID']), csv_path)


def get_easiest_graph(csv_path: str = INPUT_CSV) -> Dict:
    """Load graph with largest Δ_min (easiest)."""
    df = pd.read_csv(csv_path)
    easiest_idx = df['Delta_min'].idxmax()
    return load_graph_from_csv(int(df.loc[easiest_idx, 'Graph_ID']), csv_path)


def get_example_graph(name: str) -> Dict:
    """Load predefined example graph."""
    if name not in EXAMPLES:
        raise ValueError(f"Example '{name}' not found. Available: {list(EXAMPLES.keys())}")
    
    ex = EXAMPLES[name]
    return {
        'graph_id': None,
        'n_qubits': ex['n_qubits'],
        'edges': ex['edges'],
        'delta_min': None,
        's_at_min': None,
        'max_degeneracy': None,
        'optimal_cut': ex['optimal_cut']
    }

# =========================================================================
# QAOA EXECUTION WITH FULL DISTRIBUTION
# =========================================================================

def run_qaoa_with_distribution(edges: List[Tuple[int, int]], 
                                 n_qubits: int,
                                 p: int = 1,
                                 max_iter: int = 200) -> Dict:
    """
    Run QAOA and return full probability distribution.
    
    Returns:
        Dictionary with results including counts, optimal params, etc.
    """
    print(f"\n🔬 Running QAOA (p={p}, max_iter={max_iter}, shots={NUM_SHOTS})...")
    
    # Build cost Hamiltonian
    cost_hamiltonian = edges_to_cost_hamiltonian(edges, n_qubits)
    
    # Create QAOA ansatz circuit
    qaoa_circuit = QAOAAnsatz(cost_hamiltonian, reps=p)
    
    # Setup simulator
    backend = AerSimulator(method=SIMULATOR_METHOD)
    
    # Transpile circuit
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
        
        # Add measurements
        from qiskit import QuantumCircuit
        measured_circuit = bound_circuit.copy()
        measured_circuit.measure_all()
        
        # Run circuit
        job = backend.run(measured_circuit, shots=NUM_SHOTS, seed_simulator=RANDOM_SEED)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation value
        expectation = 0.0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            bitstring_reversed = bitstring[::-1]
            
            # Calculate energy
            energy = 0.0
            for u, v in edges:
                if bitstring_reversed[u] == bitstring_reversed[v]:
                    energy += 1.0
                else:
                    energy -= 1.0
            
            expectation += (count / total_shots) * energy
        
        cost_history.append(expectation)
        
        if iteration_count[0] % 10 == 0:
            print(f"    Iteration {iteration_count[0]:3d}: Cost = {expectation:.6f}")
        
        return expectation
    
    # Initial parameters
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
    
    print(f"    ✓ Optimization complete in {optimization_time:.2f}s")
    print(f"    Iterations: {iteration_count[0]}")
    
    # Get final distribution
    optimal_params = result.x
    bound_circuit_final = transpiled_circuit.assign_parameters(optimal_params)
    
    from qiskit import QuantumCircuit
    measured_circuit_final = bound_circuit_final.copy()
    measured_circuit_final.measure_all()
    
    job_final = backend.run(measured_circuit_final, shots=NUM_SHOTS, seed_simulator=RANDOM_SEED)
    result_final = job_final.result()
    counts = result_final.get_counts()
    
    # Convert to probabilities and reverse bitstrings (Qiskit little-endian)
    total_shots = sum(counts.values())
    prob_dist = {}
    for bitstring, count in counts.items():
        bitstring_reversed = bitstring[::-1]
        prob_dist[bitstring_reversed] = count / total_shots
    
    # Find most probable bitstring
    most_probable_bitstring = max(prob_dist, key=prob_dist.get)
    most_probable_prob = prob_dist[most_probable_bitstring]
    most_probable_cut = evaluate_cut_value(most_probable_bitstring, edges)
    
    # Calculate expected cut value (correct approximation ratio)
    # Convert prob_dist to counts for the function
    counts_from_prob = {bs: int(prob * NUM_SHOTS) for bs, prob in prob_dist.items()}
    expected_cut = calculate_expected_cut(counts_from_prob, edges)
    
    # Best measured (same as most probable for now)
    best_bitstring = most_probable_bitstring
    best_cut_value = most_probable_cut
    
    return {
        'prob_dist': prob_dist,
        'best_bitstring': best_bitstring,
        'best_cut_value': best_cut_value,
        'most_probable_bitstring': most_probable_bitstring,
        'most_probable_prob': most_probable_prob,
        'most_probable_cut': most_probable_cut,
        'expected_cut': expected_cut,
        'optimal_params': optimal_params,
        'num_iterations': iteration_count[0],
        'final_cost': result.fun,
        'optimization_time': optimization_time,
        'cost_history': cost_history
    }

# =========================================================================
# GRAPH VISUALIZATION
# =========================================================================

def create_graph_layout(edges: List[Tuple[int, int]], n_qubits: int, 
                        layout_type: str = 'circular') -> Dict:
    """
    Create NetworkX graph and compute layout positions.
    
    Returns:
        Dictionary with G (graph) and pos (positions)
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges)
    
    if layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout_type == 'shell':
        pos = nx.shell_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    return {'G': G, 'pos': pos}


def draw_original_graph(ax, edges: List[Tuple[int, int]], n_qubits: int, pos: Dict):
    """Draw graph without solution coloring."""
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, ax=ax, edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color='black', width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    ax.set_title(f'Original Graph: N={n_qubits} nodes, {len(edges)} edges', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')


def draw_solution_graph(ax, edges: List[Tuple[int, int]], n_qubits: int, 
                       bitstring: str, pos: Dict, optimal_cut: int):
    """Draw graph with solution coloring."""
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges)
    
    # Node colors based on bitstring
    node_colors = ['red' if bitstring[i] == '0' else 'blue' for i in range(n_qubits)]
    
    # Identify cut edges
    cut_edges = []
    non_cut_edges = []
    for u, v in edges:
        if bitstring[u] != bitstring[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))
    
    cut_value = len(cut_edges)
    
    # Draw non-cut edges (thin, gray)
    nx.draw_networkx_edges(G, pos, edgelist=non_cut_edges, 
                          edge_color='lightgray', width=1, ax=ax, style='dashed')
    
    # Draw cut edges (thick, green)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, 
                          edge_color='green', width=4, ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=500, ax=ax, edgecolors='black', linewidths=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', 
                           font_color='white', ax=ax)
    
    # Title with cut value
    title = f'QAOA Solution: Cut = {cut_value}/{optimal_cut}'
    if cut_value == optimal_cut:
        title += ' ✓ (Optimal!)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Legend
    red_patch = mpatches.Patch(color='red', label='Partition 0')
    blue_patch = mpatches.Patch(color='blue', label='Partition 1')
    green_line = mpatches.Patch(color='green', label=f'Cut edges ({cut_value})')
    ax.legend(handles=[red_patch, blue_patch, green_line], loc='upper right', fontsize=10)
    
    ax.axis('off')

# =========================================================================
# DISTRIBUTION VISUALIZATION
# =========================================================================

def plot_probability_distribution(ax, prob_dist: Dict, edges: List[Tuple[int, int]], 
                                   optimal_cut: int, most_probable_bitstring: str, 
                                   top_n: int = 25):
    """Plot bar chart of top N bitstrings by probability."""
    # Sort by probability
    sorted_items = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    bitstrings = [item[0] for item in sorted_items]
    probabilities = [item[1] for item in sorted_items]
    
    # Calculate cut value for each bitstring
    cut_values = [evaluate_cut_value(bs, edges) for bs in bitstrings]
    
    # Color based on cut value (green for optimal, gradient otherwise)
    max_cut = max(cut_values)
    colors = []
    for cv in cut_values:
        if cv == optimal_cut:
            colors.append('green')
        else:
            # Red to yellow gradient
            ratio = cv / optimal_cut
            colors.append(plt.cm.RdYlGn(ratio))
    
    # Create bar chart
    x_pos = np.arange(len(bitstrings))
    bars = ax.bar(x_pos, probabilities, color=colors, edgecolor='black', linewidth=1)
    
    # Highlight most probable bitstring
    for i, bs in enumerate(bitstrings):
        if bs == most_probable_bitstring:
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3)
            # Add star marker
            ax.text(i, probabilities[i] + 0.0005, '★', 
                    ha='center', fontsize=16, color='gold')
    
    # Add value labels on bars
    for i, (bar, prob, cv) in enumerate(zip(bars, probabilities, cut_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}\ncut={cv}',
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Bitstrings (sorted by probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Probability Distribution (Top {top_n} Bitstrings)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bitstrings, rotation=90, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Add info text
    total_prob = sum(probabilities)
    unique_count = len(prob_dist)
    optimal_prob = sum(prob for bs, prob in prob_dist.items() 
                      if evaluate_cut_value(bs, edges) == optimal_cut)
    
    info_text = f'Total unique: {unique_count} | Top {top_n}: {total_prob:.1%} | Optimal prob: {optimal_prob:.1%}'
    ax.text(0.5, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_cut_value_histogram(ax, prob_dist: Dict, edges: List[Tuple[int, int]], 
                              optimal_cut: int):
    """Plot histogram of cut values across all measured bitstrings."""
    # Calculate cut value for each bitstring (weighted by probability)
    cut_value_probs = {}
    for bitstring, prob in prob_dist.items():
        cut_val = evaluate_cut_value(bitstring, edges)
        if cut_val not in cut_value_probs:
            cut_value_probs[cut_val] = 0.0
        cut_value_probs[cut_val] += prob
    
    # Sort by cut value
    cut_values = sorted(cut_value_probs.keys())
    probabilities = [cut_value_probs[cv] for cv in cut_values]
    
    # Color bars
    colors = ['green' if cv == optimal_cut else 'steelblue' for cv in cut_values]
    
    # Create bar chart
    bars = ax.bar(cut_values, probabilities, color=colors, edgecolor='black', 
                  linewidth=1.5, width=0.8)
    
    # Add value labels
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Cut Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cut Value Distribution', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(cut_values)
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight optimal
    ax.axvline(optimal_cut, color='green', linestyle='--', linewidth=2, 
               label=f'Optimal cut = {optimal_cut}')
    ax.legend(loc='upper left', fontsize=11)


def add_metrics_panel(ax, graph_info: Dict, qaoa_results: Dict):
    """Add text panel with metrics summary."""
    ax.axis('off')
    
    # Build text
    lines = []
    lines.append("=" * 40)
    lines.append("GRAPH PROPERTIES")
    lines.append("=" * 40)
    lines.append(f"N (qubits):        {graph_info['n_qubits']}")
    lines.append(f"Edges:             {len(graph_info['edges'])}")
    
    if graph_info.get('delta_min') is not None:
        lines.append(f"Δ_min:             {graph_info['delta_min']:.6f}")
        lines.append(f"s at min:          {graph_info['s_at_min']:.3f}")
        lines.append(f"Max degeneracy:    {graph_info['max_degeneracy']}")
    
    lines.append(f"Optimal cut:       {graph_info['optimal_cut']}")
    
    lines.append("")
    lines.append("=" * 40)
    lines.append("QAOA RESULTS")
    lines.append("=" * 40)
    lines.append(f"p layers:          {P_LAYERS}")
    lines.append("")
    lines.append("Most Probable Bitstring:")
    lines.append(f"  {qaoa_results['most_probable_bitstring']}")
    lines.append(f"  Probability:     {qaoa_results['most_probable_prob']:.4f}")
    lines.append(f"                   ({qaoa_results['most_probable_prob']*100:.2f}%)")
    lines.append(f"  Cut value:       {qaoa_results['most_probable_cut']}/{graph_info['optimal_cut']}")
    lines.append("")
    lines.append("Expected Performance:")
    lines.append(f"  Expected cut:    {qaoa_results['expected_cut']:.4f}")
    
    approx_ratio = qaoa_results['expected_cut'] / graph_info['optimal_cut']
    lines.append(f"  Approximation:   {approx_ratio:.6f}")
    lines.append("")
    lines.append(f"Optimizer iters:   {qaoa_results['num_iterations']}")
    lines.append(f"Optimization time: {qaoa_results['optimization_time']:.2f}s")
    lines.append(f"Final cost:        {qaoa_results['final_cost']:.6f}")
    
    lines.append("")
    lines.append("=" * 40)
    lines.append("DISTRIBUTION ANALYSIS")
    lines.append("=" * 40)
    
    unique_count = len(qaoa_results['prob_dist'])
    lines.append(f"Unique bitstrings: {unique_count}")
    
    # Calculate entropy
    probs = list(qaoa_results['prob_dist'].values())
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    lines.append(f"Entropy:           {entropy:.3f} bits")
    
    # Optimal solution probability
    optimal_prob = sum(prob for bs, prob in qaoa_results['prob_dist'].items() 
                      if evaluate_cut_value(bs, graph_info['edges']) == graph_info['optimal_cut'])
    lines.append(f"Optimal sol prob:  {optimal_prob:.4f}")
    
    # Display text
    text = '\n'.join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# =========================================================================
# MAIN VISUALIZATION FUNCTION
# =========================================================================

def visualize_qaoa(graph_info: Dict):
    """
    Main function: run QAOA and create complete visualization.
    
    Args:
        graph_info: Dictionary with graph properties (from load functions)
    """
    print("=" * 70)
    print("  QAOA SINGLE GRAPH VISUALIZATION")
    print("=" * 70)
    
    # Print graph info
    if graph_info.get('graph_id') is not None:
        print(f"\n📊 Graph #{graph_info['graph_id']}")
    else:
        print(f"\n📊 Custom Graph")
    
    print(f"   N = {graph_info['n_qubits']} qubits")
    print(f"   Edges = {len(graph_info['edges'])}")
    if graph_info.get('delta_min') is not None:
        print(f"   Δ_min = {graph_info['delta_min']:.6f}")
    print(f"   Optimal cut = {graph_info['optimal_cut']}")
    
    # Run QAOA
    qaoa_results = run_qaoa_with_distribution(
        graph_info['edges'],
        graph_info['n_qubits'],
        p=P_LAYERS,
        max_iter=MAX_OPTIMIZER_ITERATIONS
    )
    
    # Print results
    print(f"\n📈 QAOA Results:")
    print(f"   Most probable bitstring: {qaoa_results['most_probable_bitstring']}")
    print(f"   Most probable probability: {qaoa_results['most_probable_prob']:.4f} ({qaoa_results['most_probable_prob']*100:.2f}%)")
    print(f"   Most probable cut: {qaoa_results['most_probable_cut']}/{graph_info['optimal_cut']}")
    print(f"   Expected cut: {qaoa_results['expected_cut']:.4f}")
    approx_ratio = qaoa_results['expected_cut'] / graph_info['optimal_cut']
    print(f"   Approximation ratio: {approx_ratio:.6f}")
    
    # Create visualization
    print(f"\n🎨 Generating visualization...")
    
    # Compute graph layout
    layout_info = create_graph_layout(graph_info['edges'], graph_info['n_qubits'], 
                                      GRAPH_LAYOUT)
    pos = layout_info['pos']
    
    # Create figure with subplots
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Original graph
    ax2 = fig.add_subplot(gs[1, 0])  # Solution graph
    ax3 = fig.add_subplot(gs[:, 1])  # Probability distribution
    ax4 = fig.add_subplot(gs[0, 2])  # Cut value histogram
    ax5 = fig.add_subplot(gs[1, 2])  # Metrics panel
    
    # Draw all panels
    draw_original_graph(ax1, graph_info['edges'], graph_info['n_qubits'], pos)
    draw_solution_graph(ax2, graph_info['edges'], graph_info['n_qubits'], 
                       qaoa_results['most_probable_bitstring'], pos, graph_info['optimal_cut'])
    plot_probability_distribution(ax3, qaoa_results['prob_dist'], 
                                  graph_info['edges'], graph_info['optimal_cut'],
                                  qaoa_results['most_probable_bitstring'],
                                  top_n=NUM_TOP_BITSTRINGS)
    plot_cut_value_histogram(ax4, qaoa_results['prob_dist'], 
                            graph_info['edges'], graph_info['optimal_cut'])
    add_metrics_panel(ax5, graph_info, qaoa_results)
    
    # Overall title
    if graph_info.get('graph_id') is not None:
        suptitle = f'QAOA Visualization: Graph #{graph_info["graph_id"]} (N={graph_info["n_qubits"]}, p={P_LAYERS})'
    else:
        suptitle = f'QAOA Visualization: Custom Graph (N={graph_info["n_qubits"]}, p={P_LAYERS})'
    
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    if SAVE_FIGURE:
        if graph_info.get('graph_id') is not None:
            filename = f"{OUTPUT_DIR}QAOA_demo_graph{graph_info['graph_id']}.png"
        else:
            filename = f"{OUTPUT_DIR}QAOA_demo_custom.png"
        
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        print(f"   ✓ Saved: {filename}")
    
    print(f"\n" + "=" * 70)
    print(f"✅ Visualization complete!")
    print(f"=" * 70)
    
    plt.show()

# =========================================================================
# COMMAND-LINE INTERFACE
# =========================================================================

def main():
    """Main entry point with command-line argument parsing."""
    import sys
    
    # Parse arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == '--hardest':
            print("Loading hardest graph (smallest Δ_min)...")
            graph_info = get_hardest_graph()
        elif arg == '--easiest':
            print("Loading easiest graph (largest Δ_min)...")
            graph_info = get_easiest_graph()
        elif arg == '--example':
            if len(sys.argv) < 3:
                print("Usage: --example <name>")
                print(f"Available examples: {list(EXAMPLES.keys())}")
                return
            example_name = sys.argv[2]
            print(f"Loading example: {example_name}...")
            graph_info = get_example_graph(example_name)
        elif arg == '--graph_id':
            if len(sys.argv) < 3:
                print("Usage: --graph_id <ID>")
                return
            graph_id = int(sys.argv[2])
            print(f"Loading Graph #{graph_id}...")
            graph_info = load_graph_from_csv(graph_id)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage:")
            print("  python3 visualize_qaoa_single_graph.py                # Default")
            print("  python3 visualize_qaoa_single_graph.py --graph_id 5")
            print("  python3 visualize_qaoa_single_graph.py --hardest")
            print("  python3 visualize_qaoa_single_graph.py --easiest")
            print("  python3 visualize_qaoa_single_graph.py --example triangle")
            return
    else:
        # Default: load graph with smallest Δ_min
        print(f"Loading default (Graph #{DEFAULT_GRAPH_ID})...")
        graph_info = load_graph_from_csv(DEFAULT_GRAPH_ID)
    
    # Run visualization
    visualize_qaoa(graph_info)


if __name__ == "__main__":
    main()

