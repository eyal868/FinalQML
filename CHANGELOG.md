# Changelog

All notable changes to the FinalQML project are documented here, grouped by milestone.

---

## [2026-02-24] Documentation Consolidation

- Consolidated 6 documentation files into 2: `README.md` (user guide) and `docs/paper_context.md` (paper mapping)
- Added architecture diagram to README
- Created this CHANGELOG
- Deleted `.cursor/Context/` files (content folded into README and paper_context)

## [2026-02-10] Output System Refactor

- Introduced dual-save output system: repo (`outputs/`) + Desktop (`~/Desktop/FinalQML_Outputs/YYYY-MM-DD/`)
- Created `output_config.py` central module (`get_run_dirs()`, `save_file()`, `save_dual()`, `save_run_info()`)
- Renamed all output files to consistent scheme (e.g. `spectral_gap_3reg_N12_k2.csv`)
- Merged `DataOutputs/` into `outputs/` and deleted it
- Added `.gitignore` rule: CSVs committed, PNGs gitignored under `outputs/`
- Refactored 12 scripts to use `output_config.py`

## [2026-01-04 – 2026-01-06] Phase 3: Weighted Graphs

- Added weighted Max-Cut support: `compute_weighted_optimal_cut()`, weighted sparse Hamiltonians
- New scripts: `weighted_gap_analysis.py`, `plot_weighted_spectrum.py`
- Created `sample_delta_uniform.py` to uniformly sample 100 N=16 graphs across gap bins
- Branch: `cleaning-up-old-gap-method`

## [2025-12-27 – 2026-01-03] Phase 2: Sparse Methods & Parallel Processing

- Refactored `aqc_spectral_utils.py` to use sparse Hamiltonians (Lanczos/eigsh) — O(N × 2^N) memory vs O(4^N) dense
- Added parallel processing via `multiprocessing.Pool` — 8–14× speedup on M3 Max
- Created unified pipeline runner `run_qaoa_pipeline.py` with CLI arguments
- Added SCD binary parser for GENREG graph files
- Enabled N=14 (~1–2 hrs parallel) and N=16 spectral gap computation
- Merged via PR #2 from `improving-gap-computation`

## [2025-11-25 – 2025-12-10] Phase 1: Optimization Improvements

- Implemented heuristic initialization (γ ∈ [0.1, π/4], β ∈ [0.1, π/2])
- Implemented warm-start (reuse p−1 optimal params for p)
- Implemented multi-start (3 attempts at p ≥ 7, early stop if ratio ≥ 0.95)
- Data retention improved from 53% to 76% (filter-out rate: 47% → 24%)
- Merged via PR #1 from `Optimization-Improvments`

## [2025-10 – 2025-11] Initial Implementation

- Spectral gap analysis for N=10 (19 graphs, M=200) and N=12 (85 graphs, M=20)
- QAOA p-sweep across p=1–10 with COBYLA optimizer
- Monotonicity filter (`filter_qaoa_monotonic.py`)
- Visualization scripts: `plot_p_sweep_ratio_vs_gap.py`, `plot_p_star_vs_gap.py`, `plot_delta_vs_degeneracy.py`, `plot_example_spectrum.py`, `visualize_qaoa_single_graph.py`
- ASC graph parser in `aqc_spectral_utils.py`
- Initial `README.md`
