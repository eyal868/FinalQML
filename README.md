# QAOA Performance and Spectral Gap Analysis

Analysis of the relationship between QAOA performance and minimum spectral gap (Δ_min) for the Max-Cut problem on 3-regular graphs. This codebase enables:
- Computation of minimum spectral gaps for adiabatic quantum computing (AQC) 
- QAOA performance evaluation across different circuit depths
- Correlation analysis between spectral gap and QAOA metrics
- Comprehensive visualization tools

## Quick Start

### Installation
```bash
pip3 install -r requirements.txt
```

### Basic Workflow

**Option A: Unified Pipeline (Recommended)**
```bash
# Run complete analysis in one command
python3 run_qaoa_pipeline.py --input outputs/Delta_min_3_regular_N12_sparse_k2_final.csv

# With parallel processing (8-14x faster on multi-core)
python3 run_qaoa_pipeline.py --input outputs/Delta_min_3_regular_N12_sparse_k2_final.csv --workers 8
```

**Option B: Step-by-Step**
```bash
# 1. Compute spectral gaps for graphs
python3 spectral_gap_analysis.py

# 2. Run QAOA performance sweep
python3 qaoa_analysis.py

# 3. Filter optimization artifacts
python3 filter_qaoa_monotonic.py

# 4. Visualize results
python3 plot_p_sweep_ratio_vs_gap.py outputs/QAOA_p_sweep_N12_p1to10.csv
python3 plot_p_star_vs_gap.py outputs/QAOA_p_sweep_N12_p1to10.csv
```

---

## Complete Workflow

### Unified Pipeline (Recommended)

Run the entire QAOA analysis workflow with a single command:

```bash
python3 run_qaoa_pipeline.py --input outputs/Delta_min_3_regular_N12_sparse_k2_final.csv --output-dir outputs/results/
```

**All CLI Options:**
```bash
python3 run_qaoa_pipeline.py \
    --input outputs/Delta_min_3_regular_N12_sparse_k2_final.csv \
    --output-dir outputs/results/ \
    --degeneracy 2 \          # Filter to graphs with degeneracy=2
    --p-min 1 --p-max 10 \    # QAOA depth range
    --max-iter 500 \          # Optimizer iterations
    --shots 10000 \           # QAOA measurement shots
    --parallel \              # Enable parallel (default)
    --workers 8               # Number of CPU cores (default: all)
```

**Skip Options:**
- `--skip-qaoa --qaoa-results path/to/existing.csv`: Use existing QAOA results
- `--skip-filter`: Skip monotonic filtering
- `--skip-plots`: Skip plot generation

**Outputs:**
- `QAOA_p_sweep_N{N}_p{min}to{max}.csv`: Raw QAOA results
- `QAOA_p_sweep_N{N}_p{min}to{max}_filtered.csv`: Filtered results
- `p_sweep_ratio_vs_gap_N{N}.png`: Approximation ratio plots
- `p_star_vs_gap_N{N}.png`: Minimum depth plots

---

### Individual Scripts

#### 1. Spectral Gap Analysis

Compute minimum spectral gap (Δ_min) along the AQC path `H(s) = (1-s)H_mixer + s·H_problem` using sparse eigensolvers (Lanczos) with Brent's optimization.

**Run with default configuration:**
```bash
python3 spectral_gap_analysis.py
```

**Configure by editing the `CONFIG` dictionary at the top of the file:**
```python
CONFIG = {
    'N_values': [12],                # Graph sizes to process
    'graphs_per_N': {                # Graph selection per N
        12: None,                    # None = all graphs
    },
    'target_degeneracy': 2,          # Filter to graphs with this degeneracy
    's_bounds': (0.01, 0.99),        # Optimization bounds for s
    'xtol': 1e-4,                    # Optimization tolerance
    'output_suffix': ''              # Optional suffix for output filename
}
```

**Output:** CSV file with auto-generated name, e.g. `outputs/Delta_min_3_regular_N10_12_res100.csv`

**Columns:**
- `N`: Number of qubits
- `Graph_ID`: Graph identifier
- `Delta_min`: Minimum spectral gap
- `s_at_min`: Location where minimum occurs
- `Max_degeneracy`: Ground state degeneracy
- `Max_cut_value`: Optimal Max-Cut value
- `Edges`: Graph edge list

---

### 2. QAOA Performance Sweep

Run QAOA across multiple depths (p=1,2,3,...) and measure approximation ratios.

**Configure by editing parameters at the top of `qaoa_analysis.py`:**

```python
# Input: Spectral gap CSV from step 1
INPUT_CSV = 'outputs/Delta_min_3_regular_N10_res200.csv'

# QAOA depth sweep
P_VALUES_TO_TEST = list(range(1, 11))  # Test p=1 through p=10

# Optimizer settings
MAX_OPTIMIZER_ITERATIONS = 500
OPTIMIZER_METHOD = 'COBYLA'
NUM_SHOTS = 10000

# Output filename
OUTPUT_FILENAME = 'old_outputs/QAOA_p_sweep_N10_p1to10.csv'
```

**Run:**
```bash
python3 qaoa_analysis.py
```

**Output:** CSV with columns for each depth:
- `p1_approx_ratio`, `p2_approx_ratio`, ..., `p10_approx_ratio`
- `p1_iterations`, `p2_iterations`, ...
- Plus all columns from input spectral gap CSV

**Performance (Sequential):**
- N=10: ~2s per graph (19 graphs total, ~40s)
- N=12: ~8s per graph (85 graphs total, ~11 min)

**Performance (Parallel with 8 workers):**
- N=10: ~5s total (~8x speedup)
- N=12: ~1.5 min total (~8x speedup)
- N=14: ~1-2 hours (vs ~10-12 hours sequential)

---

### 3. Data Filtering (Remove Optimization Artifacts)

Apply monotonicity filter to remove classical optimization failures where `ratio(p) < ratio(p-1)`.

**Process all QAOA sweep files:**
```bash
python3 filter_qaoa_monotonic.py
```

**Process specific file(s):**
```bash
python3 filter_qaoa_monotonic.py outputs/QAOA_p_sweep_N10_p1to10.csv
python3 filter_qaoa_monotonic.py file1.csv file2.csv file3.csv
```

**Output:** Creates `*_filtered.csv` versions of each input file.

**What it does:** Marks values as NaN when QAOA performance decreases (indicating optimizer got stuck, not quantum algorithm limitation).

---

### 4. Visualization & Analysis

#### 4.1 Approximation Ratio vs Spectral Gap (All Depths)

Plot correlation between Δ_min and approximation ratio for each p value.

```bash
# Unfiltered data
python3 plot_p_sweep_ratio_vs_gap.py outputs/QAOA_p_sweep_N10_p1to10.csv

# Filtered data (recommended)
python3 plot_p_sweep_ratio_vs_gap.py outputs/QAOA_p_sweep_N10_p1to10_filtered.csv
```

**Output:** Multi-panel plot showing scatter plots, trend lines, and correlation coefficients for each depth.

#### 4.2 Minimum Depth (p*) vs Spectral Gap

Analyze how many layers are needed to reach different approximation ratio thresholds.

```bash
python3 plot_p_star_vs_gap.py outputs/QAOA_p_sweep_N10_p1to10.csv
```

**Thresholds analyzed:** 0.75, 0.80, 0.85, 0.90, 0.95

**Output:** Plot showing p* (minimum depth required) vs Δ_min for each threshold.

#### 4.3 Spectral Gap vs Ground State Degeneracy

Explore relationship between gap and degeneracy.

```bash
python3 plot_delta_vs_degeneracy.py
```

**Configuration:** Edit `csv_file` path inside the script:

```python
csv_file = "DataOutputs/Delta_min_3_regular_N12_res20.csv"
```

#### 4.4 Single Graph Visualization

Interactive visualization of QAOA behavior on individual graphs.

```bash
# Default graph (smallest Δ_min)
python3 visualize_qaoa_single_graph.py

# Specific graph by ID
python3 visualize_qaoa_single_graph.py --graph_id 5

# Hardest/easiest graphs
python3 visualize_qaoa_single_graph.py --hardest
python3 visualize_qaoa_single_graph.py --easiest

# Simple example
python3 visualize_qaoa_single_graph.py --example triangle
```

**Configuration:** Edit parameters at top of file:
```python
INPUT_CSV = 'outputs/Delta_min_3_regular_N12_res20.csv'
P_LAYERS = 1  # QAOA depth
MAX_OPTIMIZER_ITERATIONS = 200
```

#### 4.5 Spectral Gap Visualization

Plot the full spectral gap evolution along the AQC path.

```bash
python3 plot_example_spectrum.py
```

**Configuration:** Edit inside the script to select specific graphs by ID.

---

## Data Files

### Graph Input Formats

**GENREG enumeration data (graphs_rawdata/):**
- `10_3_3.asc` - 19 graphs, N=10, 3-regular, girth 3 (text format)
- `12_3_3.asc` - 85 graphs, N=12, 3-regular, girth 3 (text format)
- `12_3_3.scd` - 85 graphs, N=12, 3-regular, girth 3 (binary format)

Both `.asc` and `.scd` formats are supported. The codebase auto-detects format from file extension.

### Output Files

**Spectral gap results:**
- Format: `Delta_min_{regularity}_N{size}_res{resolution}{suffix}.csv`
- Example: `Delta_min_3_regular_N10_res200.csv`

**QAOA sweep results:**
- Format: `QAOA_p_sweep_N{size}_p{min}to{max}{suffix}.csv`
- Example: `QAOA_p_sweep_N12_p1to10_deg_2_only.csv`

**Filtered results:**
- Format: Original filename + `_filtered.csv`
- Example: `QAOA_p_sweep_N10_p1to10_filtered.csv`
---

## Project Structure

```
FinalQML/
├── run_qaoa_pipeline.py               # UNIFIED PIPELINE: Run complete workflow
│
├── spectral_gap_analysis.py           # Step 1: Compute Δ_min for graphs
├── qaoa_analysis.py                   # Step 2: QAOA performance sweep (parallel)
├── filter_qaoa_monotonic.py           # Step 3: Filter optimization artifacts
│
├── plot_p_sweep_ratio_vs_gap.py       # Visualize: ratio vs gap across depths
├── plot_p_star_vs_gap.py              # Visualize: minimum depth analysis
├── plot_delta_vs_degeneracy.py        # Visualize: gap vs degeneracy
├── plot_example_spectrum.py           # Visualize: spectral gap evolution
├── visualize_qaoa_single_graph.py     # Visualize: individual graph behavior
│
├── aqc_spectral_utils.py              # Utilities: Graph I/O, Hamiltonians
├── test_scd_parser.py                 # Test: SCD file format validation
│
├── graphs_rawdata/                    # Input: GENREG graph data
│   ├── 10_3_3.asc
│   ├── 12_3_3.asc
│   └── 12_3_3.scd
│
├── outputs/                           # Output: All CSV and PNG results
│
├── requirements.txt                   # Dependencies
├── README.md                          # This file
```

---

**Last Updated:** January 3, 2026