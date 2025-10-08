# QAOA Analysis on 3-Regular Graphs

## Overview

This module analyzes QAOA performance on Max-Cut problems and correlates the results with spectral gap (Δ_min) values computed in the first part of the project.

## Quick Start

### 1. Run QAOA on 3-node validation example
```bash
python3 qaoa_analysis.py test
```

### 2. Run QAOA on all graphs from spectral gap data
```bash
python3 qaoa_analysis.py
```

### 3. Visualize correlations
```bash
python3 plot_qaoa_correlation.py
```

## Configuration

Edit parameters at the top of `qaoa_analysis.py`:

```python
# Input/Output
INPUT_CSV = 'outputs/Delta_min_3_regular_N10_res200.csv'  # Spectral gap data
OUTPUT_FILENAME = 'outputs/QAOA_results_N10_p1.csv'

# QAOA parameters
P_LAYERS = 1                      # Number of QAOA layers (p)
MAX_OPTIMIZER_ITERATIONS = 200    # Max classical optimizer iterations
OPTIMIZER_METHOD = 'COBYLA'       # Classical optimizer
NUM_SHOTS = 10000                 # Measurement shots
RANDOM_SEED = 42                  # For reproducibility

# Simulation
SIMULATOR_METHOD = 'statevector'  # Noiseless simulation
```

## Results Summary (N=10, p=1)

### Key Findings

**✅ All 19 graphs solved to optimality (approximation ratio = 1.0)**

This indicates that QAOA with p=1 is sufficient to solve Max-Cut on all 3-regular graphs with N=10 nodes.

### Statistics

| Metric | Mean | Std Dev |
|--------|------|---------|
| Δ_min | 0.943 | 0.282 |
| Approximation ratio | 1.000 | 0.000 |
| Optimizer iterations | 31.0 | 2.7 |
| Optimization time | 0.417s | 0.038s |

### Correlations with Δ_min

| QAOA Metric | Pearson r | p-value | Significance |
|-------------|-----------|---------|--------------|
| Approximation ratio | N/A | N/A | No variance (all optimal) |
| Optimizer iterations | -0.080 | 0.744 | Not significant |
| Optimization time | -0.047 | 0.847 | Not significant |
| **Final cost** | **-0.651** | **0.003** | ✅ **Significant!** |

**Key insight:** Graphs with larger Δ_min have more negative final costs (lower energy), suggesting better convergence quality.

## Output Files

### QAOA Results CSV

Columns:
- `N`: Number of qubits
- `Graph_ID`: Graph identifier (matches spectral gap data)
- `Delta_min`: Minimum spectral gap (from spectral analysis)
- `s_at_min`: Location of minimum gap
- `Max_degeneracy`: Ground state degeneracy
- `Optimal_cut`: Known optimal Max-Cut value
- `p_layers`: QAOA depth
- `QAOA_cut_value`: Cut value found by QAOA
- `Approximation_ratio`: QAOA_cut / Optimal_cut
- `Optimizer_iterations`: Number of classical optimization steps
- `Final_cost`: Final energy value
- `Optimization_time`: Time in seconds
- `Best_bitstring`: Binary solution string

### Visualization

`QAOA_correlation_N10_p1.png` contains 4 subplots:
1. **Approximation Ratio vs Δ_min** - Shows solution quality
2. **Convergence Speed vs Δ_min** - Optimizer iterations needed
3. **Optimization Time vs Δ_min** - Computational cost
4. **Final Cost vs Δ_min** - Energy landscape correlation ⭐

## Future Experiments

Since N=10, p=1 solved all instances optimally, consider:

### 1. **Increase problem size**
```python
# Run on N=12 graphs (85 graphs from GENREG file)
INPUT_CSV = 'outputs/Delta_min_3_regular_N12_res20.csv'
```

### 2. **Test multiple QAOA depths**

Create a loop to test p=1, 2, 3, 5:
```python
for p in [1, 2, 3, 5]:
    P_LAYERS = p
    OUTPUT_FILENAME = f'outputs/QAOA_results_N10_p{p}.csv'
    # Run analysis...
```

### 3. **Reduce max iterations** (create harder scenarios)
```python
MAX_OPTIMIZER_ITERATIONS = 50  # Force early stopping
```

### 4. **Different graph types**
- Erdős-Rényi random graphs
- Random geometric graphs
- Less regular structures

### 5. **Add noise models**
```python
from qiskit_aer.noise import NoiseModel
# Add realistic hardware noise
```

## Implementation Details

### Cost Hamiltonian

Max-Cut cost Hamiltonian:
```
H_cost = ∑_{(i,j)∈E} Z_i Z_j
```

Implemented as `SparsePauliOp` in Qiskit.

### QAOA Circuit

Uses `QAOAAnsatz` from Qiskit:
- Alternating cost and mixer layers
- Standard mixer: H_mixer = ∑_i X_i
- Parameters: 2p values (γ and β for each layer)

### Classical Optimization

- **Method:** COBYLA (Constrained Optimization BY Linear Approximation)
- **Initial params:** Random uniform in [0, 2π]
- **Cost function:** Expectation value ⟨H_cost⟩

### Simulation Backend

- **Backend:** `AerSimulator` with `method='statevector'`
- **Noiseless:** No decoherence or gate errors
- **Shots:** 10,000 measurements per circuit evaluation

## Validation

The 3-node triangle test validates the implementation:

**Input:**
- Graph: Triangle (3 nodes, 3 edges)
- Optimal cut: 2 edges

**Result:**
- Best bitstring: `001` (node 2 in one partition, nodes 0,1 in other)
- Cut value: 2/2 = optimal
- Approximation ratio: 1.0 ✅
- Iterations: 19
- Time: 0.07s

## Performance

| N | Graphs | Avg time/graph | Total time (p=1) |
|---|--------|----------------|------------------|
| 10 | 19 | 0.47s | ~9s |
| 12 | 85 | ~1.5s | ~2 min (estimated) |
| 14 | ~500 | ~5s | ~40 min (estimated) |

**Note:** Times scale with N (circuit size) and number of optimizer iterations.

## File Structure

```
FinalQML/
├── qaoa_analysis.py              # Main QAOA implementation
├── plot_qaoa_correlation.py      # Correlation visualization
├── QAOA_README.md                # This file
└── outputs/
    ├── QAOA_results_N10_p1.csv   # QAOA performance data
    └── QAOA_correlation_N10_p1.png  # Correlation plots
```

## Theoretical Background

### QAOA-AQC Connection

QAOA can be viewed as discretized adiabatic quantum computing:
- **AQC:** Continuous time evolution with H(t)
- **QAOA:** Discrete alternating unitaries with fixed time steps
- **Hypothesis:** Graphs with small Δ_min (hard for AQC) should also be hard for QAOA

### Expected Relationship

**Theoretical prediction:**
- Smaller Δ_min → Harder problem
- Harder problem → Lower QAOA approximation ratio (for fixed p)

**Observed (N=10, p=1):**
- All graphs solved optimally
- No correlation with iterations/time
- **But:** Significant correlation with final cost energy

**Interpretation:**
- N=10 instances too easy for p=1 QAOA
- Need larger N or more constrained optimization to see difficulty separation

## Troubleshooting

### Q: Getting import errors for qiskit?
```bash
pip3 install qiskit qiskit-aer
```

### Q: QAOA taking too long?
Reduce `NUM_SHOTS` (e.g., 1000 instead of 10000) for faster evaluation at cost of noisier results.

### Q: Want to test specific graph?
```python
from qaoa_analysis import run_qaoa
edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3), (2,4)]
result = run_qaoa(edges, n_qubits=5, p=1, max_iter=200)
print(f"Cut value: {result['best_cut_value']}")
```

### Q: How to change optimizer?
```python
OPTIMIZER_METHOD = 'SLSQP'  # Try: COBYLA, SLSQP, Powell, Nelder-Mead
```

## References

1. **QAOA Original Paper:** Farhi & Gutmann, "A Quantum Approximate Optimization Algorithm" (2014)
2. **IBM Qiskit Tutorial:** [Quantum Approximate Optimization Algorithm](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm)
3. **Max-Cut Problem:** Classic NP-hard combinatorial optimization
4. **Spectral Gap Theory:** Connection between Δ_min and algorithm runtime

---

**Last Updated:** October 2025  
**Status:** Working implementation, validated on N=10 graphs

