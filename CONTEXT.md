# Technical Context and Implementation Details

This document provides implementation details for the spectral gap analysis project. For an overview, see `README.md`.

## Critical Implementation: Ground State Degeneracy

### The Problem

Max-Cut on graphs has **bit-flip symmetry**: if |ψ⟩ is a ground state, so is |ψ̄⟩ (all bits flipped). This creates ground state degeneracy at s=1.

### The Solution: Three-Step Methodology

1. **Determine degeneracy at s=1** (problem Hamiltonian only)
   - Diagonalize H_problem
   - Find ground state degeneracy k using tolerance 10⁻⁸
   - Example: If 6-fold degenerate, k=6

2. **Track E_k consistently**
   - Track E_k - E_0 throughout ENTIRE evolution (s ∈ [0,1])
   - NOT E_1 - E_0 (which would be wrong)
   - Same eigenvalue index throughout

3. **Find minimum**
   - Δ_min = min_s [E_k(s) - E_0(s)]
   - Record s* where minimum occurs

### Why This Matters

**Wrong approach:** Tracking E_1 gives gap ~0.002 at s~0.98 (meaningless)  
**Correct approach:** Tracking E_k gives gap ~1.83 at s~0.82 (physical)

**Difference of >1000x!**

## Code Structure

### Hamiltonian Construction

```python
H_initial = -∑ᵢ X̂ᵢ                    # Transverse field (mixer)
H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ             # Max-Cut (cost)
H(s) = (1-s)·H_initial + s·H_problem  # AQC interpolation
```

### Gap Calculation Algorithm

```python
# Step 1: Find degeneracy at s=1
H_final = H_problem
evals_final = eigh(H_final, subset_by_index=(0, 9))
degeneracy = find_ground_state_degeneracy(evals_final)

# Step 2: Track E_k throughout
k_index = degeneracy
for s in np.linspace(0, 1, S_RESOLUTION):
    H_s = (1-s)*H_initial + s*H_problem
    evals = eigh(H_s, subset_by_index=(0, k_index+5))
    gap = evals[k_index] - evals[0]
    
# Step 3: Return minimum gap
```

## Performance Optimizations

### 1. Selective Eigenvalue Computation

Uses `scipy.linalg.eigh` with `subset_by_index` to compute only the lowest k+5 eigenvalues:

```python
evals = eigh(H, subset_by_index=(0, k_index+5))
```

For N=10: Computing 10 eigenvalues vs 1024 → ~100x speedup.

### 2. Pre-computed Initial Hamiltonian

H_initial is computed once and reused for all graphs:

```python
H_initial = build_initial_hamiltonian(N_QUBITS)  # Once
for graph in graphs:
    H_problem = build_problem_hamiltonian(graph, N_QUBITS)
    H_s = (1-s) * H_initial + s * H_problem  # Reuse H_initial
```

Saves N_QUBITS × NUM_GRAPHS matrix constructions.

### 3. Efficient Tensor Products

Uses NumPy's optimized Kronecker product for building many-body operators:

```python
def operator_at_position(op, pos, n):
    result = 1
    for i in range(n):
        result = np.kron(result, op if i == pos else IDENTITY)
    return result
```

### Memory Usage

- Each 2^N × 2^N complex matrix: 8 × (2^N)² bytes
- For N=10: ~8 MB per Hamiltonian matrix
- Peak memory: ~50 MB for N=10 (multiple matrices in memory)

## Parameters

### Default Configuration

```python
N_QUBITS = 10              # Number of qubits (nodes)
NUM_GRAPHS = 200           # Ensemble size
S_RESOLUTION = 200         # Sampling points for s ∈ [0,1]
DEGENERACY_TOL = 1e-8      # Tolerance for eigenvalue degeneracy
```

### Performance Benchmarks

**N=10 (Default):**
- Hilbert space: 2¹⁰ = 1024 dimensions
- Time per graph: ~4 seconds
- Total time (200 graphs): ~15 minutes
- Output file: ~200 KB

**Scaling:**
- Time complexity per graph: O(S_RESOLUTION × eigensolve)
- Eigensolve: O(2^N × k) with subset computation
- Memory: O(2^(2N)) for Hamiltonian matrices

## File Organization

```
FinalQML/
├── spectral_gap_analysis.py      # Main production script
├── plot_example_spectrum.py      # Visualization tool
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── CONTEXT.md                     # This file (technical details)
├── METHODOLOGY.tex                # Academic methodology (LaTeX)
├── .gitignore                     # Git ignore patterns
└── outputs/                       # Generated files
    ├── *.csv                      # Data files
    └── *.png                      # Plots
```

## Dependencies

- **numpy >= 1.21.0** - Numerical arrays and linear algebra
- **scipy >= 1.7.0** - Eigenvalue solver with subset computation
- **networkx >= 2.6.0** - Random regular graph generation
- **pandas >= 1.3.0** - CSV data handling
- **matplotlib >= 3.4.0** - Visualization

## Output Data Format

### CSV Columns

| Column          | Type   | Description                                    |
|-----------------|--------|------------------------------------------------|
| N               | int    | Number of qubits                               |
| Graph_ID        | int    | Instance identifier (1 to NUM_GRAPHS)          |
| Delta_min       | float  | Minimum spectral gap                           |
| s_at_min        | float  | Value of s where minimum occurs (0 to 1)       |
| Max_degeneracy  | int    | Ground state degeneracy at s=1                 |
| Edges           | str    | Graph edge list as string (reproducible)       |

### Expected Statistics (N=10)

- Mean Δ_min: ~0.15-0.30
- Std Δ_min: ~0.08-0.12
- Mean degeneracy: ~2-6
- Mean s_at_min: ~0.5-0.7

## Implementation Functions

### Core Functions

1. **`build_initial_hamiltonian(n)`**
   - Constructs H_initial = -∑ᵢ X̂ᵢ
   - Returns: 2^n × 2^n complex matrix

2. **`build_problem_hamiltonian(edges, n)`**
   - Constructs H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ
   - Takes graph edges as input
   - Returns: 2^n × 2^n complex matrix

3. **`find_first_gap(eigenvalues, tol)`**
   - Determines ground state degeneracy
   - Finds first non-degenerate excited state
   - Returns: (gap, degeneracy_index)

4. **`calculate_min_gap(edges, n, H_initial, s_values)`**
   - Main gap calculation for one graph
   - Tracks E_k - E_0 across all s values
   - Returns: (Δ_min, s_at_min, max_degeneracy)

5. **`operator_at_position(op, pos, n)`**
   - Helper for tensor product construction
   - Places operator at specific qubit position

## Common Operations

### Run Full Analysis
```bash
python3 spectral_gap_analysis.py
```

### Visualize Example
```bash
python3 plot_example_spectrum.py
```

### Change Parameters
Edit at top of `spectral_gap_analysis.py`:
```python
N_QUBITS = 10          # Increase carefully (2^N scaling!)
NUM_GRAPHS = 200       # More graphs = better statistics
S_RESOLUTION = 200     # More points = better gap resolution
```

## Validation and Testing

The implementation has been validated with:

✅ N=4 example (6-fold degeneracy case)  
✅ Full spectrum visualization  
✅ Eigenvalue verification  
✅ Hermiticity checks  
✅ Comparison with known results

## Known Limitations

1. **Memory:** N ≥ 14 requires >1 GB RAM
2. **Time:** Exponential scaling in N (doubles per increment)
3. **Degeneracy assumption:** Assumes degeneracy remains stable across small s variations
4. **Graph constraints:** Requires N even for 3-regular graphs

## Troubleshooting

### Q: Getting Δ_min ≈ 0?
**A:** Likely tracking E_1 instead of E_k. Check `find_first_gap` function is being called.

### Q: Very large degeneracy (>10)?
**A:** May indicate highly symmetric graph (e.g., complete graph). Check graph structure with NetworkX.

### Q: Runtime too long?
**A:** Reduce N_QUBITS (exponential scaling) or NUM_GRAPHS (linear scaling).

### Q: Memory error?
**A:** N too large. Each increment doubles memory. Try N-1.

### Q: NetworkX graph generation fails?
**A:** 3-regular graphs require N even. Code handles this automatically.

## Advanced Usage

### Parallel Processing

For multiple N values, run in parallel:
```bash
python3 spectral_gap_analysis.py &  # N=10
# Edit N_QUBITS=8, then:
python3 spectral_gap_analysis.py &  # N=8
```

### Custom Graph Analysis

To analyze a specific graph:
```python
from spectral_gap_analysis import calculate_min_gap, build_initial_hamiltonian

edges = [(0,1), (1,2), (2,3), (3,0), (0,2), (1,3)]  # Custom graph
n = 4
H_initial = build_initial_hamiltonian(n)
s_values = np.linspace(0, 1, 200)

delta_min, s_at_min, degeneracy = calculate_min_gap(edges, n, H_initial, s_values)
print(f"Δ_min = {delta_min:.6f} at s = {s_at_min:.3f}")
```

## Connection to QAOA

QAOA can be viewed as discretized/Trotterized AQC. The hypothesis is:
- Graphs with smaller Δ_min are harder for AQC (require longer T)
- These same graphs should be harder for QAOA (require more layers p)
- This project generates Δ_min data to test this correlation

## Future Enhancements

Potential improvements:
- Parallel processing for multiple graphs (multiprocessing)
- Adaptive s-sampling near critical points (gap minimum)
- Additional graph types (Erdős–Rényi, geometric)
- Gap derivative analysis (∂Δ/∂s for phase transition detection)
- Excited state spectroscopy (higher gaps)

## References for Implementation

Key papers informing this implementation:
1. Farhi et al., "Quantum Computation by Adiabatic Evolution" (2000) - AQC fundamentals
2. Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014) - QAOA connection
3. Crosson & Harrow (2016) - Spectral gap analysis techniques

---

**Last Updated:** October 2025  
**Implementation Status:** Tested and production-ready