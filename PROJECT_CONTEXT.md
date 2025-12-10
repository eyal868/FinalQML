# QAOA Spectral Gap Research Project - Baseline Context

> **🔬 ACTIVE RESEARCH PROJECT**: This codebase supports an ongoing research project investigating the relationship between QAOA performance and the minimum spectral gap in Adiabatic Quantum Computing. The project is actively being expanded with additional graph sizes, optimization methods, and analysis techniques.

**Last Updated**: November 25, 2025  
**Project Repository**: https://github.com/eyal868/FinalQML

---

## Table of Contents

1. [Paper Summary](#paper-summary)
2. [Key Findings](#key-findings)
3. [Paper Structure Index](#paper-structure-index)
4. [Figure Reference Index](#figure-reference-index)
5. [Codebase Structure](#codebase-structure)
6. [Current Datasets and Results](#current-datasets-and-results)
7. [Expansion Guidelines](#expansion-guidelines)
8. [Development Standards](#development-standards)
9. [Documentation Cross-Reference](#documentation-cross-reference)

---

## Paper Summary

**Title**: "QML Final Project - Project 5: Numerical Analyses of The Relation Between QAOA Performance and The Minimum Spectral Gap"

**Author**: Eyal Chai-Ezra

**Abstract**: This work investigates whether the QAOA required circuit depth (p) depends on the minimum spectral gap (g_min) in the same way that Adiabatic Quantum Computing runtime scales as T ∝ 1/g_min². Through numerical analysis of 3-regular Max-Cut graphs (N=10, 12) with controls for ground state degeneracy, we discovered counter-intuitive results: instances with smaller g_min (AQC-hard) often achieved **higher** approximation ratios at fixed depths. Correspondingly, instances with larger g_min (AQC-easy) appeared to require **more** layers to reach target accuracy. We conclude that g_min is **not a reliable predictor** for QAOA performance, which is governed by a more complex interplay between the quantum ansatz and classical parameter optimization.

---

## Key Findings

### 1. Counter-Intuitive Negative Correlation
- At fixed shallow depths (p=1-4), **negative correlation** between g_min and approximation ratio
- Instances with **smaller g_min** (traditionally "AQC-hard") perform **better** in QAOA
- Trend observed consistently in degeneracy-controlled subsets (k=2, k=4)

### 2. Inverted Critical Depth Relationship
- Analysis of minimum depth (p*) required to reach target approximation ratios shows **positive correlation** with g_min
- Larger spectral gaps require **more** QAOA layers—opposite of AQC expectation
- **Caveat**: Heavily confounded by classical optimization difficulty

### 3. Ground State Degeneracy is Critical
- Controlling for ground state degeneracy is **essential**
- Without degeneracy control, positive correlations appear (Figure: N12 all degeneracies)
- With degeneracy control, reveals the true negative/absent correlation
- Degeneracy itself correlates with g_min, acting as confounding variable

### 4. Classical Optimization as Major Confound
- **~45% of data filtered** due to non-monotonic performance (optimization failures)
- COBYLA optimizer struggles with high-dimensional parameter landscapes at large p
- Optimization difficulty likely masks true quantum p* relationship
- Performance degradation at p≥7 attributed to local minima, not quantum limitations

### 5. No Simple AQC→QAOA Translation
- The AQC scaling T ∝ 1/g_min² does **not** translate to QAOA depth requirements
- QAOA's variational nature and diabatic effects allow circumventing small gaps
- g_min alone is insufficient to predict QAOA resource requirements

---

## Paper Structure Index

### Section 1: Introduction
- **1.1 MaxCut Problem**: NP-hard problem, Goemans-Williamson 0.878 approximation
- **1.2 Adiabatic Quantum Computing**: Adiabatic theorem, H(s) evolution, gap dependency
- **1.3 QAOA**: Ansatz structure, discretized/Trotterized AQC, Suzuki-Trotter equivalence

### Section 2: Research Questions
- **Q1**: Relationship between g_min and approximation ratio at fixed depth p
- **Q2**: Relationship between g_min and critical depth p* for fixed approximation ratio
- **Prediction**: No simple relationship expected due to diabatic advantages and variational flexibility

### Section 3: Methodology
- **3.1.1 Graph Selection**: 3-regular graphs from GENREG database
  - N=10: 19 graphs
  - N=12: 85 graphs
- **3.1.2 Calculating g_min**: Diagonalization at M points along s∈[0,1]
  - N=10: M=200 resolution
  - N=12: M=20 resolution (computational constraint)
- **3.1.3 Ground State Degeneracy**: Proper handling of k-fold degeneracy
  - Gap calculation: g(s) = E_k(s) - E_0(s) where k is degeneracy at s=1
- **3.1.4 Degeneracy Bias**: Control for degeneracy correlation with g_min
- **3.2 QAOA Implementation**: 
  - p = 1 to 10 layers
  - COBYLA optimizer, max 500 iterations
  - Statevector simulator (noiseless)
  - Approximation ratio = ⟨H_C⟩ / C_max

### Section 4: Results and Discussion
- **4.1 Depth Sweep Results**: 
  - Monotonicity filter removes ~45% of data
  - N=12, k=2: Strong negative correlation across most p values
  - N=12, k=4: Weak correlations, absent positive trend
  - Decreasing valid data points at high p indicates optimization difficulty
- **4.2 Critical Depth (p*) Results**:
  - Positive correlation: larger g_min → larger p* required
  - Thresholds analyzed: 0.75, 0.80, 0.85, 0.90, 0.95
  - Confounded by classical optimization failures inflating measured p*

### Section 5: Conclusions
- g_min is **not a reliable predictor** of QAOA performance
- Expected positive correlation is **absent**
- Complex picture: inverse trend at low p (unclear mechanism), noise-dominated at high p
- True QAOA cost governed by quantum ansatz + classical parameter tractability

### Section 6: Future Research and Improvements
1. **Generalizability Testing**: Extend beyond MaxCut (Vertex Cover, SAT, Number Partitioning)
2. **High-Precision Gap Calculation**: Adaptive search, binary search methods
3. **Classical Optimization Enhancement**: Warm-start, multi-start, basin-hopping

### Section 7: Supplementary Material
- Code repository link
- Additional plots for N=10, non-filtered data
- Optimization failure examples

---

## Figure Reference Index

Map of LaTeX figure references to actual files in repository:

| LaTeX Reference | File in Repo | Description |
|-----------------|--------------|-------------|
| `example_full_spectrum_N12_graph18.pdf` | `DataOutputs/example_full_spectrum_N12_graph18.pdf` | Energy eigenvalue evolution for Graph 18 showing proper gap calculation with degeneracy |
| `example_full_spectrum_N12_graph44.pdf` | `DataOutputs/example_full_spectrum_N12_graph44.pdf` | High degeneracy example (k=36) |
| `Delta_min_3_regular_N12_res20_delta_vs_degeneracy.pdf` | `DataOutputs/Delta_min_3_regular_N12_res20_delta_vs_degeneracy.pdf` | Correlation between g_min and ground state degeneracy |
| `p_sweep_ratio_vs_gap_N12_p1to10_deg_2_only_filtered.pdf` | `DataOutputs/p_sweep_ratio_vs_gap_N12_p1to10_deg_2_only_filtered.png` | **Main Result**: Negative correlation at k=2 (filtered) |
| `p_sweep_ratio_vs_gap_N12_p1to10_deg_4_only_filtered.pdf` | `DataOutputs/p_sweep_ratio_vs_gap_N12_p1to10_deg_4_only_filtered.png` | k=4 results showing weak/absent correlation (filtered) |
| `p_star_vs_gap_N12_p1to10_deg_2_only_filtered.png` | `DataOutputs/p_star_vs_gap_N12_p1to10_deg_2_only_filtered.png` | Critical depth analysis for k=2 |
| `p_star_vs_gap_N12_p1to10_deg_4_only_filtered.png` | `DataOutputs/p_star_vs_gap_N12_p1to10_deg_4_only_filtered.png` | Critical depth analysis for k=4 |
| `optimization_failure_examples.pdf` | `DataOutputs/optimization_failure_examples.png` | Representative cases of classical optimization failures |
| `p_sweep_ratio_vs_gap_N12_p1to10.pdf` | `DataOutputs/p_sweep_ratio_vs_gap_N12_p1to10.png` | Supplementary: All degeneracies combined (shows confounding) |
| `p_sweep_ratio_vs_gap_N12_p1to10_deg_2_only.pdf` | `DataOutputs/p_sweep_ratio_vs_gap_N12_p1to10_deg_2_only.png` | Supplementary: k=2 unfiltered |
| `p_sweep_ratio_vs_gap_N12_p1to10_deg_4_only.pdf` | `DataOutputs/p_sweep_ratio_vs_gap_N12_p1to10_deg_4_only.png` | Supplementary: k=4 unfiltered |
| `p_sweep_ratio_vs_gap_N10_p1to10_deg_2_only_filtered.pdf` | `DataOutputs/p_sweep_ratio_vs_gap_N10_p1to10_deg_2_only_filtered.png` | Supplementary: N=10 results |

**Note**: Paper references `.pdf` files but actual figures in `DataOutputs/` are `.png` format (except spectrum plots which exist as both).

---

## Codebase Structure

### Core Analysis Pipeline

#### 1. Spectral Gap Calculation
**Script**: `spectral_gap_analysis.py`

- **Purpose**: Compute minimum spectral gap (g_min) along AQC evolution path
- **Input**: Graph data from `graphs_rawdata/` (ASC or SCD format)
- **Configuration**: 
  - `N_values`: Graph sizes to process [10, 12]
  - `S_resolution`: Sampling points along s∈[0,1] (20-200)
  - `graphs_per_N`: Subset selection (None=all, int=first N, range=specific)
  - `k_vals_check`: Max degeneracy threshold (skip if exceeded)
  - `degree`: Graph regularity (3, 4, or 5)
- **Output**: CSV with columns: N, Graph_ID, Delta_min, s_at_min, Max_degeneracy, Max_cut_value, Edges
- **Key Features**:
  - Proper degeneracy handling (tracks E_k - E_0, not E_1 - E_0)
  - Supports 3-regular, 4-regular, 5-regular graphs
  - Auto-generates output filename from configuration

#### 2. QAOA Performance Analysis
**Script**: `qaoa_analysis.py`

- **Purpose**: Run QAOA p-sweep and measure approximation ratios
- **Input**: Spectral gap CSV from step 1
- **Configuration**:
  - `P_VALUES_TO_TEST`: Circuit depths to test [1, 2, ..., 10]
  - `MAX_OPTIMIZER_ITERATIONS`: 500
  - `OPTIMIZER_METHOD`: COBYLA
  - `NUM_SHOTS`: 10000
  - `SIMULATOR_METHOD`: statevector (noiseless)
- **Output**: CSV with p1_approx_ratio, p2_approx_ratio, ..., p1_iterations, p2_iterations, ...
- **Performance**: N=10 ~2s/graph, N=12 ~8s/graph
- **Key Features**:
  - Qiskit-based implementation (QAOAAnsatz, AerSimulator)
  - Tracks approximation ratio, iterations, cost function value
  - Random parameter initialization (seed=42)

#### 3. Data Filtering
**Script**: `filter_qaoa_monotonic.py`

- **Purpose**: Remove optimization artifacts where ratio(p) < ratio(p-1)
- **Input**: QAOA sweep CSV files
- **Output**: `*_filtered.csv` versions
- **Effect**: Removes ~45% of data in typical runs
- **Rationale**: Classical optimizer failures should not be interpreted as quantum algorithm limitations

#### 4. Visualization Tools

| Script | Purpose | Key Output |
|--------|---------|------------|
| `plot_p_sweep_ratio_vs_gap.py` | Correlation between g_min and ratio across all p | Multi-panel scatter plots with trend lines |
| `plot_p_star_vs_gap.py` | Minimum depth (p*) analysis for thresholds | p* vs g_min for 0.75, 0.80, 0.85, 0.90, 0.95 |
| `plot_delta_vs_degeneracy.py` | Gap vs degeneracy relationship | Scatter plot showing correlation |
| `plot_example_spectrum.py` | Energy eigenvalue evolution | Full spectrum along s∈[0,1] |
| `visualize_qaoa_single_graph.py` | Single graph interactive visualization | Graph structure + QAOA performance |
| `plot_optimization_failure_examples.py` | Publication figure of optimizer failures | Representative failure cases |

### Utility Modules

**Script**: `aqc_spectral_utils.py`

Core functions for quantum analysis:
- **Hamiltonian Construction**: `build_H_initial()`, `build_H_problem()`, `get_aqc_hamiltonian()`
- **Gap Finding**: `find_first_gap()`, `find_min_gap_with_degeneracy()`
- **Graph I/O**: 
  - `parse_asc_file()` - Text format GENREG graphs
  - `parse_scd_file()` - Binary format GENREG graphs (with differential compression)
  - `load_graphs_from_file()` - Format-agnostic loader
  - `extract_graph_params()` - Extract n, k, girth from filename
  - `adjacency_to_edges()`, `shortcode_to_edges()` - Conversion utilities

**Script**: `test_scd_parser.py`

- Validation suite for SCD binary format parser
- Compares ASC vs SCD parsing for correctness
- 100% match rate confirmed for 12_3_3 (85 graphs)

### Data Organization

#### Input Data (`graphs_rawdata/`)
- `10_3_3.asc` - 19 graphs, N=10, 3-regular, girth 3
- `10_4_3.asc` - N=10, 4-regular, girth 3
- `10_5_3.asc` - N=10, 5-regular, girth 3
- `12_3_3.asc` - 85 graphs, N=12, 3-regular, girth 3 (text)
- `12_3_3.scd` - 85 graphs, N=12, 3-regular, girth 3 (binary)

#### Output Data
- **Primary**: `outputs/` directory
- **Published**: `DataOutputs/` directory (used in paper)

**Naming Conventions**:
- Spectral gap: `Delta_min_{degree}_regular_N{size}_res{resolution}{suffix}.csv`
- QAOA sweep: `QAOA_p_sweep_N{size}_p{min}to{max}{suffix}.csv`
- Filtered: `{original_name}_filtered.csv`
- Plots: Match CSV names with `.png` or `.pdf` extension

---

## Current Datasets and Results

### Graph Datasets

| Size | Degree | Count | Source File | Status |
|------|--------|-------|-------------|--------|
| N=10 | 3-regular | 19 | `10_3_3.asc` | ✅ Analyzed (M=200) |
| N=10 | 4-regular | ? | `10_4_3.asc` | ⚠️ File exists, not yet analyzed in paper |
| N=10 | 5-regular | ? | `10_5_3.asc` | ⚠️ File exists, not yet analyzed in paper |
| N=12 | 3-regular | 85 | `12_3_3.asc/.scd` | ✅ Analyzed (M=20) |

### Spectral Gap Results

**N=10 (19 graphs, 3-regular)**:
- Resolution: M=200 sampling points
- Gap range: [documented in CSV]
- Degeneracy distribution: [documented in CSV]

**N=12 (85 graphs, 3-regular)**:
- Resolution: M=20 sampling points  
- Gap range: g_min ∈ [0.34, 0.96]
- Degeneracy distribution:
  - k=2 (1 distinct solution): 31 graphs (36.5%)
  - k=4 (2 distinct solutions): 26 graphs (30.6%)
  - k≥6: 28 graphs (32.9%)

### QAOA Performance Results

**Configuration Used**:
- Circuit depths: p = 1, 2, 3, ..., 10
- Optimizer: COBYLA, max 500 iterations
- Simulator: Qiskit Aer statevector (noiseless)
- Random seed: 42

**Key Metrics**:
- Approximation ratio range: ~0.75 to ~0.95
- Optimization success rate: ~55% (45% filtered)
- Performance: N=10 ~40s total, N=12 ~11min total

**Published Analyses**:
- Depth sweep correlation analysis (all p values)
- Critical depth (p*) analysis (thresholds: 0.75-0.95)
- Degeneracy-controlled subsets (k=2, k=4)
- Filtered vs unfiltered comparisons

---

## Expansion Guidelines

### 🎯 PRIMARY DIRECTIVE

**This is an ACTIVE research project being expanded for publication improvement.** All development should focus on:

1. **Adding more data** (larger N, more graphs, different regularity)
2. **Improving optimization** (reducing the 45% data loss from filtering)
3. **Extending analysis** (higher p, new problems, noise modeling)

### Explicit Future Work from Paper (Section 6)

#### 1. Generalizability Testing
- [ ] Extend beyond MaxCut problem
  - [ ] Minimum Vertex Cover
  - [ ] Satisfiability (SAT) problems
  - [ ] Number Partitioning
- [ ] Diverse MaxCut instance types
  - [ ] Random graphs (not just regular)
  - [ ] Planar graphs
  - [ ] Scale-free networks
  - [ ] Geometric graphs

#### 2. High-Precision Gap Calculation
- [ ] Implement adaptive search for g_min location
  - Binary search around local minima
  - Higher resolution near s_min
- [ ] Efficient methods for larger N
  - Sparse matrix techniques
  - Krylov subspace methods for partial diagonalization
- [ ] Validation: Compare M=20 vs M=200 for N=12

#### 3. Classical Optimization Enhancement
- [ ] **Warm-Start Initialization** (Priority 1)
  - Use optimal parameters from p-1 as starting point for p
  - Expected: 20-30% reduction in failures
  - Implementation: Extend previous params with small perturbation
- [ ] **Multi-Start Optimization** (Priority 2)
  - Run 3-5 random initializations, keep best
  - Expected: 50-70% reduction in failures
  - Adaptive: Stop early if target ratio reached
- [ ] **Advanced Optimizers**
  - Basin-hopping for global search
  - Differential evolution (genetic algorithm)
  - L-BFGS-B with gradients
- [ ] **Heuristic Initialization**
  - Problem-specific parameter ranges
  - Literature-based starting points for MaxCut
- [ ] **Comparative Study**
  - Test all methods on subset of graphs
  - Document: ratio achieved, time cost, variance

### Technical Expansion Directions

#### Larger System Sizes
- [ ] N=14 graphs (if available in GENREG)
- [ ] N=16 graphs (computational feasibility check)
- [ ] N=18+ (requires sparse methods, GPU acceleration)
- **Challenges**: 
  - Memory: 2^N × 2^N Hamiltonian matrices
  - Time: Diagonalization O(2^3N), QAOA simulation O(2^N)

#### Additional Graph Types
- [x] 3-regular (complete)
- [ ] 4-regular (data exists: `10_4_3.asc`, needs analysis)
- [ ] 5-regular (data exists: `10_5_3.asc`, needs analysis)
- [ ] Mixed regularity comparison study
- [ ] Non-regular graphs (requires new data source)

#### Extended Circuit Depths
- [x] p=1 to 10 (complete)
- [ ] p=11 to 20 (some data exists for N=12)
- [ ] p>20 for shallow-depth instances
- **Note**: Optimization difficulty increases with p

#### Noise Modeling
- [ ] Depolarizing noise channels
- [ ] Gate error rates from real hardware (IBM, Rigetti)
- [ ] Measurement errors
- [ ] Impact on g_min correlation
- **Goal**: NISQ-relevant analysis

### Data Quality Standards

**For all new analyses, maintain**:

1. **Degeneracy Control**
   - Always report and control for ground state degeneracy
   - Separate analyses for k=2, k=4, k≥6 subsets
   - Document degeneracy distribution in dataset

2. **Monotonicity Filtering**
   - Apply filter to remove optimization artifacts
   - Report: % data filtered, per-graph statistics
   - Create both filtered and unfiltered versions

3. **Complete Metadata**
   - Track: N, Graph_ID, g_min, s_min, degeneracy, max_cut_value
   - Record: optimizer iterations, convergence status
   - Document: all hyperparameters, random seeds

4. **Reproducibility**
   - Fixed random seed (default: 42)
   - Version control all configuration changes
   - CSV outputs include all parameters in filename

5. **Visualization Standards**
   - Publication-quality figures (300 DPI)
   - Color-coded by correlation strength
   - Include statistics (r, p-value, n)
   - Both PNG (paper) and PDF (vector) formats

---

## Development Standards

### Code Conventions

**Configuration Management**:
- All user-configurable parameters at top of file in `CONFIG` dict
- Auto-generate output filenames from configuration
- Document all options with inline comments

**Function Documentation**:
- Docstrings for all public functions
- Include: Purpose, Args, Returns, Notes
- Example usage for complex functions

**Error Handling**:
- Validate input files exist before processing
- Check CSV format compatibility
- Graceful degradation for missing columns

**Performance**:
- Print progress updates for long-running tasks
- Report timing information (per-graph, total)
- Use vectorized NumPy operations where possible

### Testing Requirements

**Before committing new analysis**:
1. Run on small test case (1-2 graphs)
2. Verify output CSV format matches expectations
3. Check plot generation succeeds
4. Validate against known results (if available)

**Validation scripts**:
- `test_scd_parser.py` - Graph I/O correctness
- Manual spot-checks for spectral gap values
- Compare filtered vs unfiltered statistics

### Git Workflow

**Current branch**: `Optimization-Improvments` (note: typo in branch name)

**Untracked files** (consider adding):
- `.gitignore`
- `This_Project_Paper_QAOA_spectral_gap.pdf`
- `docs/` directory

**Commit guidelines**:
- Descriptive messages: "Add N=14 spectral gap analysis"
- Separate: data generation, analysis, visualization
- Tag: releases/milestones (e.g., v1.0-paper-submission)

### Output Organization

**Primary outputs** → `outputs/`  
**Published/paper outputs** → `DataOutputs/`  
**Intermediate/debug** → Local files (add to .gitignore)

**Backup strategy**:
- CSV files are version-controlled
- Large data files (>10MB) should be archived separately
- Generated figures can be regenerated from CSVs

---

## Documentation Cross-Reference

### Project Documentation Files

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | User guide, workflow, quick start | ✅ Complete |
| `PROJECT_CONTEXT.md` | **This file** - Baseline context and expansion guide | ✅ Current |
| `QAOA_SPECTRAL_GAP_ANALYSIS.md` | Detailed statistical results and methodology | ⚠️ Predates paper, may have different data |
| `QAOA_OPTIMIZATION_CHALLENGES.md` | Deep dive on classical optimizer problems | ✅ Complete, highly detailed |
| `SCD_IMPLEMENTATION_SUMMARY.md` | Binary graph format support | ✅ Complete |
| `OPTIMIZATION_FAILURE_EXAMPLES.md` | Publication figure documentation | ✅ Complete |

### Quick Navigation

**Getting Started**:
- New user? → Start with `README.md`
- Understanding optimizer issues? → `QAOA_OPTIMIZATION_CHALLENGES.md`
- Need current paper summary? → This file (Paper Summary section)

**Running Analyses**:
- Spectral gap calculation → `README.md` Section 1
- QAOA performance sweep → `README.md` Section 2
- Filtering and visualization → `README.md` Sections 3-4

**Extending the Project**:
- What to expand? → This file (Expansion Guidelines section)
- How to add graphs? → `README.md` + `aqc_spectral_utils.py` docstrings
- Optimization improvements? → `QAOA_OPTIMIZATION_CHALLENGES.md`

**Understanding Results**:
- Current findings → This file (Key Findings section)
- Statistical details → `QAOA_SPECTRAL_GAP_ANALYSIS.md`
- Figure interpretations → This file (Figure Reference Index)

### External Resources

- **GENREG Database**: https://www.mathe2.uni-bayreuth.de/markus/reggraphs.html
- **Project Repository**: https://github.com/eyal868/FinalQML
- **Qiskit Documentation**: https://qiskit.org/documentation/

---

## Changelog

**November 25, 2025**: Initial creation of PROJECT_CONTEXT.md baseline document
- Indexed full paper structure from LaTeX source
- Mapped all figures to repository files
- Documented complete codebase structure
- Established expansion guidelines and future work checklist
- Created cross-reference index to all existing documentation

---

## Notes for Future Development

### When Adding New Graph Sizes

1. Update `GENREG_FILES` dict in `spectral_gap_analysis.py`
2. Consider computational limits:
   - N=14: ~16K basis, ~30min per graph
   - N=16: ~65K basis, several hours per graph
3. May need to reduce S_resolution for larger N
4. Update this document with new dataset statistics

### When Implementing New Optimizers

1. Create comparison study first (5-10 graphs)
2. Document: success rate, time cost, final ratios
3. Update `qaoa_analysis.py` configuration options
4. Add to `QAOA_OPTIMIZATION_CHALLENGES.md`
5. Regenerate filtered datasets if improvement is significant

### When Extending to New Problems

1. Define problem Hamiltonian in `aqc_spectral_utils.py`
2. Create separate analysis scripts (don't modify MaxCut ones)
3. New output directory structure
4. Compare: g_min correlation, degeneracy effects, optimization difficulty
5. Document findings in new markdown file

### Before Paper Submission

- [ ] Regenerate all figures with final data
- [ ] Verify all figure files match LaTeX references
- [ ] Run full pipeline on final dataset
- [ ] Update statistics in paper if changed
- [ ] Archive exact code version used (git tag)
- [ ] Upload datasets to permanent repository (Zenodo, etc.)

---

**END OF PROJECT_CONTEXT.md**

