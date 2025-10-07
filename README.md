# Spectral Gap Analysis for Adiabatic Quantum Computing

This project analyzes the **minimum spectral gap (Δ_min)** of the Adiabatic Quantum Computing (AQC) Hamiltonian for **Max-Cut on 3-regular graphs**. The goal is to study the connection between Δ_min and QAOA performance, where AQC runtime scales as **T ∝ 1/(Δ_min)²**.

## Key Innovation: Degeneracy-Aware Gap Calculation

Max-Cut problems have **bit-flip symmetry** causing ground state degeneracy. This implementation correctly tracks **E_k - E_0** (where k = degeneracy) throughout the evolution, not just E_1 - E_0. This yields physically meaningful gaps (~1.8) instead of meaningless near-zero values (~0.002).

## Quick Start

### Installation
```bash
pip3 install -r requirements.txt
```

### Run Analysis
```bash
python3 spectral_gap_analysis.py
```
Output: `outputs/Delta_min_3_regular_N10_200graphs.csv`

### Visualize Methodology
```bash
python3 plot_example_spectrum.py
```
Output: `outputs/example_full_spectrum_N4.png`

## Configuration

Edit parameters in `spectral_gap_analysis.py`:
```python
N_QUBITS = 10          # Number of qubits (nodes in graph)
NUM_GRAPHS = 200       # Number of random graph instances
S_RESOLUTION = 200     # Sampling points for s ∈ [0, 1]
```

### Performance Guidelines

| N_QUBITS | Hilbert Space | Time per Graph | 200 Graphs |
|----------|---------------|----------------|------------|
| 6        | 64            | ~0.2s          | ~1 min     |
| 8        | 256           | ~1s            | ~3 min     |
| 10       | 1024          | ~4s            | ~15 min    |
| 12       | 4096          | ~15s           | ~50 min    |

**Warning:** N ≥ 14 requires significant RAM and time.

## Theory

### Hamiltonian Definition

The AQC Hamiltonian interpolates between:

**Initial (Mixer):**
```
H_initial = -∑ᵢ X̂ᵢ
```

**Problem (Max-Cut):**
```
H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ
```

**Time-Dependent:**
```
H(s) = (1-s)·H_initial + s·H_problem,  s ∈ [0, 1]
```

### Spectral Gap

The spectral gap at parameter s is:
```
Δ(s) = E_k(s) - E_0(s)
```

where k is the ground state degeneracy at s=1. The minimum spectral gap:
```
Δ_min = min_{s∈[0,1]} Δ(s)
```

determines the adiabatic evolution time needed to remain in the ground state.

## Output Format

CSV file with columns:
- `N`: Number of qubits
- `Graph_ID`: Instance identifier
- `Delta_min`: Minimum spectral gap
- `s_at_min`: Location where minimum occurs
- `Max_degeneracy`: Ground state degeneracy at s=1
- `Edges`: Graph edge list (for reproduction)

## Analysis Examples

### Load and Explore Data
```python
import pandas as pd

df = pd.read_csv('outputs/Delta_min_3_regular_N10_200graphs.csv')

print(f"Mean Δ_min: {df['Delta_min'].mean():.6f}")
print(f"Std Δ_min: {df['Delta_min'].std():.6f}")

# Find hardest instance
hardest = df.loc[df['Delta_min'].idxmin()]
print(f"\nHardest graph: ID {hardest['Graph_ID']}")
print(f"  Δ_min = {hardest['Delta_min']:.6f}")
```

### Recreate Specific Graph
```python
import ast
import networkx as nx

edges = ast.literal_eval(df.loc[0, 'Edges'])
G = nx.Graph()
G.add_edges_from(edges)
```

## Research Applications

1. **QAOA Comparison**: Run QAOA on same graphs, correlate performance with Δ_min
2. **Scaling Studies**: Vary N (6, 8, 10, 12) to study Δ_min(N) scaling
3. **Graph Properties**: Correlate Δ_min with clustering, girth, etc.
4. **Hardness Prediction**: Use Δ_min to predict algorithm performance

## Project Structure

```
FinalQML/
├── spectral_gap_analysis.py      # Main analysis script
├── plot_example_spectrum.py      # Visualization tool
├── requirements.txt               # Dependencies
├── README.md                      # This file (overview)
├── CONTEXT.md                     # Technical implementation details
├── METHODOLOGY.tex                # LaTeX methodology for papers
└── outputs/                       # Generated data and plots
```

## Troubleshooting

**Getting Δ_min ≈ 0?**
- Check that you're tracking E_k, not E_1. The code handles this automatically.

**Runtime too long?**
- Reduce N_QUBITS (exponential scaling) or NUM_GRAPHS.

**Memory error?**
- N too large. Each increment doubles memory usage.

**Large degeneracy (>10)?**
- May indicate highly symmetric graph. Check graph structure.

## References

- Farhi, E., et al. "Quantum Computation by Adiabatic Evolution" (2000)
- Farhi, E., et al. "A Quantum Approximate Optimization Algorithm" (2014)
- Crosson & Harrow, "Simulated quantum annealing can be exponentially faster than classical simulated annealing" (2016)

## License

This research code is provided for academic use.

---

**Last Updated:** October 2025  
**Status:** Production-ready