# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Running the Analysis

### Option 1: Python Script (Recommended for large runs)

```bash
python3 spectral_gap_analysis.py
```

This will:
- Generate 200 random 3-regular graphs with N=10 nodes
- Calculate the minimum spectral gap for each
- Save results to `Delta_min_3_regular_N10_200graphs.csv`
- Estimated runtime: 15-25 minutes

### Option 2: Jupyter Notebook (Recommended for exploration)

```bash
jupyter notebook spectral_gap_analysis.ipynb
```

Then run all cells sequentially. The notebook includes:
- Detailed explanations
- Progress tracking
- Visualization of results

### Option 3: Quick Test

To verify everything works (takes ~1 second):
```bash
python3 test_small_example.py
```

## Customizing Parameters

Edit these variables at the top of the script or notebook:

```python
N_QUBITS = 10          # Number of qubits (recommend ≤ 10 for speed)
NUM_GRAPHS = 200       # Number of graphs to analyze
S_RESOLUTION = 200     # Sampling resolution for s ∈ [0,1]
```

### Performance Guidelines

| N_QUBITS | Hilbert Space | RAM Usage | Time per Graph |
|----------|---------------|-----------|----------------|
| 6        | 64            | ~1 MB     | ~0.2 sec       |
| 8        | 256           | ~2 MB     | ~1 sec         |
| 10       | 1024          | ~10 MB    | ~4 sec         |
| 12       | 4096          | ~50 MB    | ~15 sec        |

**Warning:** N=14 or higher may take hours and require significant RAM!

## Understanding the Output

The CSV file contains:

| Column    | Description                                    |
|-----------|------------------------------------------------|
| N         | Number of qubits                               |
| Graph_ID  | Unique identifier for each graph instance      |
| Delta_min | Minimum spectral gap (smaller = harder)        |
| s_at_min  | Where along the path the minimum occurs        |
| Edges     | Graph edge list (for reproducing the instance) |

## Example Analysis Workflow

1. **Generate data** (15-25 min):
   ```bash
   python3 spectral_gap_analysis.py
   ```

2. **Load and explore** in Python:
   ```python
   import pandas as pd
   
   df = pd.read_csv('Delta_min_3_regular_N10_200graphs.csv')
   
   print(f"Mean Δ_min: {df['Delta_min'].mean():.6f}")
   print(f"Std Δ_min: {df['Delta_min'].std():.6f}")
   
   # Find hardest instance
   hardest = df.loc[df['Delta_min'].idxmin()]
   print(f"\nHardest graph: ID {hardest['Graph_ID']}")
   print(f"  Δ_min = {hardest['Delta_min']:.6f}")
   ```

3. **Recreate a specific graph**:
   ```python
   import ast
   import networkx as nx
   
   # Load edges from CSV
   edges_str = df.loc[0, 'Edges']
   edges = ast.literal_eval(edges_str)
   
   # Recreate graph
   G = nx.Graph()
   G.add_edges_from(edges)
   
   # Verify it's 3-regular
   print(f"Degrees: {dict(G.degree())}")
   ```

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
Install dependencies:
```bash
pip3 install -r requirements.txt
```

### Script is too slow
Reduce parameters:
```python
N_QUBITS = 8          # Instead of 10
NUM_GRAPHS = 50       # Instead of 200
S_RESOLUTION = 100    # Instead of 200
```

### Out of memory
Reduce N_QUBITS (each increment doubles memory usage).

### "NetworkXError: no graph with degree 3"
This is rare but can happen. The script automatically skips these and continues.

## Next Steps

After generating data:

1. **Compare with QAOA**: Run QAOA on the same graphs and correlate performance with Δ_min

2. **Study scaling**: Run multiple times with different N values (6, 8, 10) to study how Δ_min scales

3. **Graph properties**: Correlate Δ_min with graph properties (clustering coefficient, girth, etc.)

4. **Location of minimum**: Analyze the distribution of `s_at_min` values

## Citation

If you use this code in your research, please cite:
- Farhi et al., "Quantum Computation by Adiabatic Evolution" (2000)
- Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
