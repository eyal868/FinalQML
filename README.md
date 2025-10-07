# Spectral Gap Analysis for Adiabatic Quantum Computing

This project analyzes the connection between the minimum energy gap (spectral gap) of the Adiabatic Quantum Computing (AQC) Hamiltonian and the performance of quantum algorithms on random 3-regular graphs.

## Research Background

In Adiabatic Quantum Computing, the runtime scales as:

**T ∝ 1/(Δ_min)²**

where Δ_min is the minimum spectral gap of the time-dependent Hamiltonian H(s) during the evolution from s=0 to s=1.

## Project Structure

- `spectral_gap_analysis.py` - Standalone Python script for running the analysis
- `spectral_gap_analysis.ipynb` - Jupyter notebook with detailed explanations and visualizations
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Python script

```bash
python spectral_gap_analysis.py
```

### Option 2: Use the Jupyter notebook

```bash
jupyter notebook spectral_gap_analysis.ipynb
```

## Configuration

Edit the following parameters in the script or notebook:

- `N_QUBITS`: Number of qubits/nodes (default: 10)
- `NUM_GRAPHS`: Number of random graph instances (default: 200)
- `S_RESOLUTION`: Number of interpolation points (default: 200)

## Hamiltonian Definition

The AQC Hamiltonian interpolates between:

### Initial Hamiltonian (Mixer)
**H_initial = -∑ᵢ X̂ᵢ**

Transverse field that creates superposition of all computational basis states.

### Problem Hamiltonian (Max-Cut)
**H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ**

Ising Hamiltonian whose ground state encodes the maximum cut of the graph.

### Time-Dependent Hamiltonian
**H(s) = (1-s)·H_initial + s·H_problem**

where s ∈ [0, 1] is the interpolation parameter.

## Output

The script generates a CSV file: `Delta_min_3_regular_N{N}_{NUM_GRAPHS}graphs.csv`

### Columns:
- `N`: Number of qubits
- `Graph_ID`: Graph instance identifier
- `Delta_min`: Minimum spectral gap
- `s_at_min`: Value of s where minimum gap occurs
- `Edges`: Graph edges (for reproduction)

## Performance Optimization

The code uses several optimizations for computational efficiency:

1. **Selective Eigenvalue Computation**: Uses `scipy.linalg.eigh` with `subset_by_index=(0,1)` to compute only the two lowest eigenvalues instead of all 2^N eigenvalues.

2. **Pre-computed Initial Hamiltonian**: H_initial is computed once and reused for all graphs.

3. **Efficient Tensor Products**: Uses NumPy's optimized `kron` function for building many-body operators.

## Expected Runtime

For N=10 qubits and 200 graphs:
- Expected runtime: 15-25 minutes on a standard laptop
- Memory usage: ~100 MB

## Theory

### Spectral Gap
The spectral gap at each point s is:
**Δ(s) = E₁(s) - E₀(s)**

where E₀(s) is the ground state energy and E₁(s) is the first excited state energy.

### Minimum Gap
The minimum spectral gap is:
**Δ_min = min_{s∈[0,1]} Δ(s)**

This quantity determines the adiabatic evolution time needed to remain in the ground state with high probability.

## Next Steps

After generating the data, you can:
1. Analyze correlations between Δ_min and graph properties
2. Compare with QAOA performance on the same graph instances
3. Study scaling behavior for different N values
4. Investigate the distribution of s where Δ_min occurs

## References

- Farhi, E., et al. "Quantum Computation by Adiabatic Evolution" (2000)
- Farhi, E., et al. "A Quantum Approximate Optimization Algorithm" (2014)

## License

This research code is provided for academic use.
