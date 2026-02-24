# Paper Context and Research Documentation

> **Consolidated reference** for the FinalQML research project. For usage instructions, see [`README.md`](../README.md).

**Last Updated**: February 24, 2026
**Repository**: https://github.com/eyal868/FinalQML

---

## Paper Information

**Title**: "Numerical Analyses of The Relation Between QAOA Performance and The Minimum Spectral Gap"

**Author**: Eyal Chai-Ezra

**Status**: Active research project — currently being expanded

**Research Question**: Does the QAOA circuit depth (p) depend on the minimum spectral gap (g_min) in the same way that AQC runtime scales as T ∝ 1/g_min²?

**Answer**: **No** — g_min is not a reliable predictor of QAOA performance.

---

## Key Findings

1. **Negative Correlation**: At fixed shallow depths (p=1–4), instances with smaller g_min (AQC-hard) achieve **higher** approximation ratios in QAOA
2. **Inverted Depth**: Instances with larger g_min require **more** layers to reach target accuracy — opposite of AQC expectation
3. **Degeneracy Critical**: Without controlling for ground state degeneracy, positive correlations appear; with control, the true negative/absent correlation is revealed
4. **Optimization Confounds**: ~24% of data filtered after optimization improvements (down from ~45%)
5. **No Simple AQC→QAOA Translation**: QAOA's variational nature and diabatic effects allow circumventing small gaps

---

## Current Datasets

| Size | Degree | Count | Status |
|------|--------|-------|--------|
| N=10 | 3-regular | 19 | Fully analyzed |
| N=12 | 3-regular | 85 | Fully analyzed, degeneracy subsets (k=2: 31, k=4: 26) |
| N=14 | 3-regular | ~509 | Spectral gaps computed (sparse), QAOA feasible (~1–2hrs parallel) |
| N=16 | 3-regular | 100 (uniform-sampled) | Spectral gaps computed |
| Weighted | N=12 | partial | Weighted Max-Cut partially implemented |

- **Circuit depths**: p = 1 to 10
- **Optimizer**: COBYLA, max 500 iterations, statevector simulator (noiseless)
- **Initialization**: Heuristic (p=1) + warm-start (p≥2) + multi-start (p≥7)
- **Random seed**: 42

---

## Paper Structure Index

| Section | Topic | Key Details |
|---------|-------|-------------|
| 1.1 | MaxCut Problem | NP-hard, Goemans-Williamson 0.878 bound |
| 1.2 | Adiabatic Quantum Computing | Adiabatic theorem, H(s) evolution, gap dependency |
| 1.3 | QAOA | Ansatz structure, Suzuki-Trotter equivalence to AQC |
| 2 | Research Questions | Q1: g_min vs ratio at fixed p; Q2: g_min vs p* at fixed ratio |
| 3.1 | Graph Selection & Gap Calc | GENREG 3-regular graphs, sparse Lanczos + Brent optimization |
| 3.2 | QAOA Implementation | p=1–10, COBYLA, statevector, approx ratio = ⟨H_C⟩/C_max |
| 4.1 | Depth Sweep Results | Monotonicity filter, negative correlation at k=2 |
| 4.2 | Critical Depth (p*) | Positive correlation: larger g_min → larger p* |
| 5 | Conclusions | g_min is **not** a reliable predictor |
| 6 | Future Work | Generalizability, precision, optimization improvements |

---

## Code-to-Paper Mapping

### Main Analysis Scripts

| Paper Section | Script | Key Config |
|---------------|--------|------------|
| Section 3.1.2 (g_min calculation) | `spectral_gap_analysis.py` | Sparse Lanczos + Brent optimization |
| Section 3.2 (QAOA analysis) | `qaoa_analysis.py` | p=1–10, COBYLA, 500 iters, parallel |
| Section 4.1 (filtering) | `filter_qaoa_monotonic.py` | Monotonicity filter |
| Section 4.1 (depth sweep) | `plot_p_sweep_ratio_vs_gap.py` | Multi-panel correlations |
| Section 4.2 (critical depth) | `plot_p_star_vs_gap.py` | Thresholds 0.75–0.95 |

### Figure Generation

| Figure | Script | Output Location |
|--------|--------|-----------------|
| Energy Spectrum Graph 18 | `plot_example_spectrum.py` | `outputs/figures/` |
| Gap vs Degeneracy | `plot_delta_vs_degeneracy.py` | `outputs/spectral_gap/` |
| Main Result (k=2) | `plot_p_sweep_ratio_vs_gap.py` | `outputs/qaoa_unweighted/N12/` |
| p* Analysis | `plot_p_star_vs_gap.py` | `outputs/qaoa_unweighted/N12/` |
| Optimization Failures | `plot_optimization_failure_examples.py` | `outputs/figures/` |
| Weighted Spectrum | `plot_weighted_spectrum.py` | `outputs/qaoa_weighted/` |

### Figure–LaTeX Cross-Reference

| LaTeX Reference | Repo File | Description |
|-----------------|-----------|-------------|
| `example_full_spectrum_N12_graph18.pdf` | `outputs/figures/example_full_spectrum_N12_graph18.*` | Energy eigenvalue evolution, proper gap with degeneracy |
| `example_full_spectrum_N12_graph44.pdf` | `outputs/figures/example_full_spectrum_N12_graph44.*` | High degeneracy example (k=36) |
| `Delta_min_3_regular_N12_res20_delta_vs_degeneracy.pdf` | `outputs/spectral_gap/spectral_gap_3reg_N12_k2_delta_vs_degeneracy.png` | g_min vs ground state degeneracy |
| `p_sweep_ratio_vs_gap_N12_p1to10_deg_2_only_filtered.pdf` | `outputs/qaoa_unweighted/N12/p_sweep_ratio_vs_gap_*_filtered.png` | **Main Result**: negative correlation at k=2 |
| `p_sweep_ratio_vs_gap_N12_p1to10_deg_4_only_filtered.pdf` | `outputs/qaoa_unweighted/N12/p_sweep_ratio_vs_gap_*_k4_filtered.png` | k=4: weak/absent correlation |
| `p_star_vs_gap_N12_p1to10_deg_2_only_filtered.png` | `outputs/qaoa_unweighted/N12/p_star_vs_gap_*_filtered.png` | Critical depth analysis for k=2 |
| `optimization_failure_examples.pdf` | `outputs/figures/optimization_failure_examples.png` | Representative optimizer failures |

> **Note**: Paper references `.pdf` but repo figures are `.png` (except spectrum plots which may exist as both).

---

## Development Timeline

### Phase 1: Optimization Improvements (Nov 25 – Dec 10, 2025)

Branch: `Optimization-Improvments` (merged via PR #1)

- Implemented **heuristic initialization**, **warm-start** (reuse p-1 params for p), and **multi-start** (3 attempts at p≥7)
- Improved data retention from **53% to 76%** (filter-out rate: 47% → 24%)
- Heuristic init: γ ∈ [0.1, π/4], β ∈ [0.1, π/2]
- Multi-start: 3 attempts at p≥7, early stop if ratio ≥ 0.95

### Phase 2: Sparse Methods and Parallel Processing (Dec 27, 2025 – Jan 3, 2026)

Branch: `improving-gap-computation` (merged via PR #2)

- Refactored `aqc_spectral_utils.py` to use **sparse Hamiltonians** (Lanczos/eigsh) — O(N × 2^N) memory vs O(4^N) for dense
- Added **parallel processing** via `multiprocessing.Pool` — 8–14x speedup on M3 Max
- Created unified pipeline runner `run_qaoa_pipeline.py` with CLI args
- Added SCD binary parser for GENREG graph files
- Enabled N=14 (~1–2hrs) and N=16 analysis

### Phase 3: Weighted Graphs (Jan 4–6, 2026)

Branch: `cleaning-up-old-gap-method` (merged)

- Added **weighted Max-Cut** support: `compute_weighted_optimal_cut()`, weighted sparse Hamiltonians
- New scripts: `weighted_gap_analysis.py`, `plot_weighted_spectrum.py`
- Created `sample_delta_uniform.py` to uniformly sample 100 N=16 graphs across gap bins

### Output System Refactor (Feb 2026)

- Dual-save system: `outputs/` (repo) + `~/Desktop/FinalQML_Outputs/` (timestamped)
- `output_config.py` central module
- Merged `DataOutputs/` into `outputs/`, consistent naming scheme
- Refactored 12 scripts

---

## Optimization Improvements — Key Results

The classical optimization challenge (COBYLA getting stuck in local minima at high p) was a major confound. Three strategies were implemented in Phase 1:

| Strategy | When Applied | Effect |
|----------|-------------|--------|
| Heuristic initialization | p=1 | γ ∈ [0.1, π/4], β ∈ [0.1, π/2] instead of random |
| Warm-start | p≥2 | Reuse optimal params from p−1 |
| Multi-start | p≥7 | 3 random attempts, early stop if ratio ≥ 0.95 |

**Aggregate improvement:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Valid data points | 165/310 (53.2%) | 236/310 (76.1%) | **+22.9%** |
| Filtered out | 145/310 (46.8%) | 74/310 (23.9%) | **-22.9%** |

**Per-depth results (N=12, k=2, with warm-start + multi-start):**

| p | Valid/Total | Mean Ratio | ≥0.90 | ≥0.95 |
|---|-------------|------------|-------|-------|
| 7 | 24/31 | 0.9152 | 67.7% | 3.2% |
| 8 | 23/31 | 0.9233 | 64.5% | 9.7% |
| 9 | 22/31 | 0.9324 | 64.5% | 9.7% |
| 10 | 15/31 | 0.9381 | 45.2% | 12.9% |

**References**: Guerreschi & Matsuura 2019 (optimization challenges), Zhou et al. 2020 (initialization strategies), Akshay et al. 2020 (parameter concentrations).

---

## Expansion Roadmap

### Completed

- [x] Sparse eigensolvers with Brent's optimization (Phase 2)
- [x] Warm-start, multi-start, heuristic initialization (Phase 1)
- [x] Parallel processing — multiprocessing.Pool (Phase 2)
- [x] N=14 and N=16 spectral gap computation (Phase 2)
- [x] Weighted Max-Cut Hamiltonians (Phase 3)
- [x] Optimizer benchmarking script — `benchmark_optimizers.py`
- [x] Output system refactor — dual-save, consistent naming

### In Progress / Planned

- [ ] Run QAOA p-sweeps on N=14 and N=16 datasets
- [ ] Finish weighted Max-Cut pipeline integration
- [ ] Non-regular graphs (Erdős-Rényi, power-law, real-world networks)
- [ ] Noise model analysis (depolarizing channels, IBM hardware specs)
- [ ] Extend to other combinatorial problems (Vertex Cover, SAT, Number Partitioning)
- [ ] Statistical rigor: bootstrap CIs, Spearman rank correlation, effect sizes

---

## Critical Development Rules

1. **Control for degeneracy** — analyze k=2, k=4 separately
2. **Apply monotonicity filter** — remove optimization artifacts
3. **Document all parameters** — in CSV filenames and metadata
4. **Reproducibility** — fixed seed (42), version-controlled configs
5. **Publication-quality figures** — 300 DPI, statistics shown (r, p-value, n)

### Data Quality Checklist

- [ ] Degeneracy controlled
- [ ] Filtered and unfiltered versions
- [ ] Complete metadata (g_min, s_min, degeneracy, max_cut)
- [ ] Reproducible (fixed seed, documented config)
- [ ] Publication-quality figures (300 DPI, statistics shown)

---

## External Resources

- **GENREG Database**: https://www.mathe2.uni-bayreuth.de/markus/reggraphs.html
- **Repository**: https://github.com/eyal868/FinalQML
- **Qiskit Documentation**: https://qiskit.org/documentation/

