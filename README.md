# Spectral Gap Analysis for Adiabatic Quantum Computing

Analysis of minimum spectral gap (Δ_min) for the Adiabatic Quantum Computing Hamiltonian on Max-Cut problems for 3-regular graphs. Studies the connection between Δ_min and QAOA performance, where AQC runtime scales as **T ∝ 1/(Δ_min)²**.

## Key Innovation

**Degeneracy-Aware Gap Calculation**: Correctly tracks **E_k - E_0** (where k = degeneracy at s=1) throughout evolution, not just E_1 - E_0. This yields physically meaningful gaps (~1.8) instead of near-zero artifacts (~0.002).

## Quick Start

### Installation
```bash
pip3 install -r requirements.txt
```

### Run Analysis
```bash
python3 spectral_gap_analysis.py
```

Edit `CONFIG` at top of file to customize:
```python
CONFIG = {
    'N_values': [10, 12],           # Which N to process
    'S_resolution': 100,             # Sampling resolution
    'graphs_per_N': {                # Graph selection
        10: None,                    # None=all, int=first N, range/list=specific
        12: range(8, 85)            # Skip problematic graphs
    },
    'k_vals_check': 20,              # Eigenvalues to check (increase for high degeneracy)
    'output_suffix': ''              # Optional filename suffix
}
```

Output: Auto-generated CSV filename based on configuration

### Visualize
```bash
python3 plot_example_spectrum.py
```

## Theory

### Hamiltonians

**Initial (Mixer):**
```
H_initial = -∑ᵢ X̂ᵢ
```

**Problem (Max-Cut):**
```
H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ
```

**AQC Path:**
```
H(s) = (1-s)·H_initial + s·H_problem,  s ∈ [0, 1]
```

### Spectral Gap

Gap at parameter s:
```
Δ(s) = E_k(s) - E_0(s)
```
where k = ground state degeneracy at s=1.

Minimum spectral gap:
```
Δ_min = min_{s∈[0,1]} Δ(s)
```

## Configuration Examples

**Process all graphs:**
```python
CONFIG = {'N_values': [10, 12], 'S_resolution': 100, 'graphs_per_N': {10: None, 12: None}}
# Output: Delta_min_3_regular_N10_12_res100.csv
```

**Skip problematic N=12 graphs:**
```python
CONFIG = {'N_values': [12], 'S_resolution': 100, 'graphs_per_N': {12: range(8, 85)}, 'output_suffix': '_skip8'}
# Output: Delta_min_3_regular_N12_res100_skip8.csv
```

**Quick test:**
```python
CONFIG = {'N_values': [10], 'S_resolution': 20, 'graphs_per_N': {10: 5}, 'output_suffix': '_test'}
# Output: Delta_min_3_regular_N10_res20_test.csv
```

## Output Format

CSV columns:
- `N`: Number of qubits
- `Graph_ID`: Unique identifier
- `Delta_min`: Minimum spectral gap
- `s_at_min`: Location where minimum occurs
- `Max_degeneracy`: Ground state degeneracy at s=1
- `Max_cut_value`: Maximum cut value
- `Edges`: Graph edge list (for reproduction)

## Data Source

Uses complete GENREG enumeration:
- N=10: 19 graphs from `10_3_3.asc`
- N=12: 85 graphs from `12_3_3.asc`

See [GENREG](https://www.mathe2.uni-bayreuth.de/markus/reggraphs.html) for details.

## Performance

| N | Graphs | Time/Graph | Total Time |
|---|--------|------------|------------|
| 10 | 19 | ~2s | ~40s |
| 12 | 85 | ~8s | ~11min |

Resolution 100 (default). Time scales linearly with `S_resolution`.

## Project Structure

```
FinalQML/
├── spectral_gap_analysis.py      # Main analysis script
├── spectral_gap_analysis_test.py # Alternative Hamiltonian formulation
├── plot_example_spectrum.py      # Visualization tool
├── 10_3_3.asc, 12_3_3.asc        # GENREG data files
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── METHODOLOGY.tex                # LaTeX methodology for papers
```

## References

- Farhi et al., "Quantum Computation by Adiabatic Evolution" (2000)
- Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
- Crosson & Harrow, "Simulated quantum annealing..." (2016)
- GENREG: Meringer, "Fast generation of regular graphs..." (1999)

## License

Academic research use.

---

**Last Updated:** October 2025  
**Status:** Production-ready
