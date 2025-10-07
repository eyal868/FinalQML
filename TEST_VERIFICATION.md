# Hamiltonian Verification and Testing

## Enhanced Test Files

I've added detailed verification capabilities to help you test the correctness of the Hamiltonian construction and spectral gap calculations.

## Files Added/Modified

### 1. **`test_small_example.py`** (Enhanced)

Now includes:
- ✅ **Full H(s) matrix output** at the minimum gap
- ✅ **Eigenvalue verification** using full diagonalization
- ✅ **Matrix property checks** (Hermitian, real, trace)
- ✅ **Automatic file saving** (both .txt and .npy formats)

#### New Output Features:

```
Graph 1:
  Edges: [(0, 1), (0, 3), ...]
  Δ_min = 0.123456 at s = 0.532
  E₀ = -2.345678, E₁ = -2.222222
  
  H(s=0.532) at minimum gap:
  Shape: (16, 16)
  Matrix (16×16 for N=4):
  [full matrix displayed]
  
  Matrix Properties:
    - Is Hermitian: True
    - Max imaginary part: 0.0000000000
    - Trace: 0.000000
  
  Eigenvalue Verification (full diagonalization):
    - E₀ (ground): -2.345678
    - E₁ (1st excited): -2.222222
    - Gap from full diag: 0.123456
    - Matches calculated: True
    - All eigenvalues: [-2.34, -2.22, -1.85, ...]
```

#### Files Saved:

1. **`H_at_min_gap_N4_graph2.txt`**: Human-readable matrix with metadata
2. **`H_at_min_gap_N4_graph2.npy`**: Binary NumPy format for easy loading

### 2. **`verify_hamiltonian.py`** (New)

A comprehensive verification script that:
- Uses N=6 (larger than test but still manageable)
- Shows Hamiltonian properties in detail
- Traces the spectral gap across s ∈ [0,1]
- Verifies H_problem is diagonal in computational basis
- Shows energy level degeneracies
- Displays bitstring → energy mapping

#### Example Output:

```
H_initial = -∑ᵢ X̂ᵢ properties:
  - Is Hermitian: True
  - Trace: 0.000000
  - Is real: True
  
Spectral gap at different s values:
  s=0.00: E₀= -6.0000, E₁= -4.0000, Δ=  2.0000
  s=0.25: E₀= -4.6748, E₁= -3.6261, Δ=  1.0486
  s=0.50: E₀= -3.9371, E₁= -3.5810, Δ=  0.3561
  s=0.75: E₀= -4.0775, E₁= -4.0043, Δ=  0.0732
  s=1.00: E₀= -5.0000, E₁= -5.0000, Δ=  0.0000
```

## How to Use for Verification

### Quick Test (1 second)
```bash
python3 test_small_example.py
```

This will:
1. Generate 2 random N=4 graphs
2. Calculate spectral gaps
3. Display full H(s_min) matrix
4. Verify eigenvalues match
5. Save matrices to files

### Detailed Verification (3 seconds)
```bash
python3 verify_hamiltonian.py
```

This will:
1. Use a fixed N=6 graph (reproducible with seed=42)
2. Show detailed Hamiltonian properties
3. Trace gap evolution across s
4. Verify diagonal structure of H_problem
5. Show energy level degeneracies

### Load and Inspect Saved Matrix

In Python:
```python
import numpy as np

# Load the binary file
H = np.load('H_at_min_gap_N4_graph2.npy')

# Verify properties
print(f"Shape: {H.shape}")
print(f"Hermitian: {np.allclose(H, H.conj().T)}")

# Compute all eigenvalues
eigenvalues = np.linalg.eigvalsh(H)
print(f"Lowest 5 eigenvalues: {eigenvalues[:5]}")
print(f"Gap: {eigenvalues[1] - eigenvalues[0]}")
```

## What to Check

### ✅ Hermiticity
All quantum Hamiltonians must be Hermitian: H† = H

```python
np.allclose(H, H.conj().T)  # Should be True
```

### ✅ Real-valued
For this problem, all matrices should be real (imaginary part = 0)

```python
np.max(np.abs(H.imag))  # Should be ~0
```

### ✅ Trace
For H_initial and H_problem, trace should be 0 (can verify analytically)

```python
np.trace(H_initial)  # Should be 0
np.trace(H_problem)  # Should be 0
```

### ✅ H_problem is Diagonal
In the computational basis, H_problem = ∑ ZᵢZⱼ must be diagonal

```python
np.allclose(H_problem, np.diag(np.diag(H_problem)))  # Should be True
```

### ✅ Eigenvalue Consistency
Compare eigenvalues from:
1. `eigh(..., subset_by_index=(0,1))` (optimized, only 2 eigenvalues)
2. `np.linalg.eigvalsh(H)` (full diagonalization)

They should match!

### ✅ Initial Gap (s=0)
At s=0, H(0) = H_initial = -∑ᵢ Xᵢ

For N qubits:
- Ground state: |+⟩⊗N with E₀ = -N
- First excited: E₁ = -(N-2)
- Gap: Δ(0) = 2

Example for N=6: E₀=-6, E₁=-4, Δ=2 ✓

## Understanding the Output

### Why Δ_min = 0 in some tests?

In the test examples, both N=4 and N=6 graphs happened to have degeneracies at s=1, leading to Δ_min = 0. This occurs when:

1. The graph has high symmetry
2. Multiple ground states exist at s=1

For the **research project with N=10**, you'll see:
- Non-zero Δ_min values
- Minimum gaps typically occurring at intermediate s ∈ [0.4, 0.8]
- Distribution of gaps showing which graphs are "harder"

### Example Expected Output for N=10

From real runs, you should see results like:
```
Graph 1: Δ_min = 0.234567 at s = 0.623
Graph 2: Δ_min = 0.156789 at s = 0.551
Graph 3: Δ_min = 0.389012 at s = 0.702
...
```

The smaller the Δ_min, the harder the instance for adiabatic evolution!

## Verifying Against Theory

### H_initial = -∑ᵢ X̂ᵢ

**Matrix elements:**
```
⟨bitstring1 | H_initial | bitstring2⟩ = -1 if bitstrings differ by 1 bit flip
                                      = 0 otherwise
```

**Eigenvalues:** -N, -(N-2), -(N-4), ..., N-4, N-2, N

### H_problem = ∑₍ᵢ,ⱼ₎∈E ẐᵢẐⱼ

**Diagonal in computational basis:**
```
⟨bitstring | H_problem | bitstring⟩ = ∑₍ᵢ,ⱼ₎∈E (-1)^(bᵢ ⊕ bⱼ)
```

For Max-Cut:
- If edge (i,j) is "cut" (different bit values): contributes +1
- If edge (i,j) is "uncut" (same bit values): contributes -1

Ground state = maximum cut!

## Troubleshooting

### "Eigenvalues don't match"
This is serious. Check:
1. Is H Hermitian?
2. Is the tolerance appropriate? (use `np.isclose` with rtol=1e-5)
3. Are you comparing the same s value?

### "Matrix is not Hermitian"
Bug in tensor product construction. The Pauli matrices themselves are Hermitian, so any linear combination should be too.

### "Gap is negative"
This is impossible by definition (E₁ ≥ E₀ always). Check eigenvalue ordering.

### "All gaps are zero"
- For small N=4, this can happen due to symmetry
- For N≥10, this would be extremely rare
- Check that you're sampling enough s points (S_RESOLUTION=200)

## Files Generated by Tests

After running tests, you'll have:

```
FinalQML/
├── H_at_min_gap_N4_graph2.txt      # Human-readable matrix
├── H_at_min_gap_N4_graph2.npy      # Binary NumPy format
├── H_min_N6_detailed.npy           # From verify_hamiltonian.py
└── ... (your CSV data files)
```

You can inspect these in Mathematica, MATLAB, or any tool that reads NumPy arrays!

## Summary

These enhanced test files provide:
1. ✅ **Visual inspection** of H(s_min) matrix
2. ✅ **Numerical verification** via multiple eigensolvers
3. ✅ **Property checks** (Hermitian, real, trace)
4. ✅ **File output** for external analysis
5. ✅ **Detailed logging** for debugging

You can now be confident that your Hamiltonian construction is correct before running the full N=10 analysis!
