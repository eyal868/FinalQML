# Branch: random-graphs-analysis — Status

## What Has Been Done

### 1. `graph_generation.py` (NEW)
Random graph generation module with:
- `generate_erdos_renyi(N, edge_prob, seed)` — Erdos-Renyi G(N,p) graphs
- `generate_random_regular(N, degree, seed)` — random regular graphs (any degree)
- `generate_watts_strogatz(N, k, rewire_prob, seed)` — small-world graphs
- `generate_random_graphs_batch(N, num_graphs, model, seed, **params)` — batch generation
- `compute_spectral_gaps_for_graphs(N, graphs, ...)` — spectral gap computation reusing existing sparse Hamiltonian builders, outputs same CSV schema as `spectral_gap_analysis.py`

### 2. Success Probability Metric in `qaoa_analysis.py` (MODIFIED)
- Added `calculate_success_probability(counts, edges, optimal_cut, weights, tol)` — computes P(measuring optimal solution), handles degeneracy naturally
- Added `optimal_cut` parameter and `success_prob` return field to:
  - `run_qaoa_single()`
  - `run_qaoa_single_quiet()`
  - `run_qaoa_multistart()` (verbose)
  - `run_qaoa_multistart_quiet()`
  - `run_qaoa()` dispatcher
- Added `p{p}_success_prob` columns to both parallel (`process_single_graph`) and sequential (`main`) output paths
- Error fallback also includes `success_prob = -1`

### 3. Weighted QAOA Pipeline in `run_qaoa_pipeline.py` (MODIFIED)
- `analyze_weighted_graph()` now passes `optimal_cut` to QAOA and records `success_prob`
- `run_qaoa_analysis_weighted()` includes `success_prob` in per-p result rows
- Added `--graph-source` flag (`genreg` or `random`) with associated args:
  - `--random-model`, `--random-N`, `--random-num-graphs`, `--random-edge-prob`, `--random-degree`, `--random-seed`
- Added `run_random_graph_spectral_analysis()` that generates random graphs + computes gaps before QAOA

### 4. `run_random_graph_analysis.py` (NEW)
Standalone combined analysis script:
- Step 1: Generate random graphs (configurable model)
- Step 2: Compute spectral gaps
- Step 3: Weighted analysis (optional, `--weighted`)
- Step 4: QAOA p-sweeps with success_prob + approx_ratio
- Outputs standard CSV format compatible with all plotting/filtering tools

### 5. Plotting Scripts (MODIFIED)
- `plot_p_sweep_ratio_vs_gap.py`: Added `metric` parameter to `plot_ratio_vs_gap()` — supports `'approx_ratio'` (default) and `'success_prob'`, with auto-fallback, correct y-axis limits, and metric-aware output filenames
- `plot_p_star_vs_gap.py`: Added `metric` parameter to `plot_p_star_vs_gap()` — supports same metrics, with appropriate default thresholds for success_prob ([0.10, 0.20, 0.30, 0.40, 0.50])

## What's Left To Do

### Testing
- [ ] Run unit tests: generate ER graphs N=8/10, verify edge lists work with Hamiltonian builders (basic test passed on this machine)
- [ ] Test `calculate_success_probability` on a known small example (4-node graph)
- [ ] Integration test: run `run_random_graph_analysis.py` with small params (N=10, 5 graphs, p=1-3)
- [ ] Run existing tests (`pytest tests/`) to make sure nothing is broken

### Remaining Work
- [ ] Test weighted + random graph combination end-to-end
- [ ] Run a real analysis: `python run_random_graph_analysis.py --model erdos_renyi --N 10 --num-graphs 20 --p-max 5`
- [ ] Run weighted analysis: `python run_random_graph_analysis.py --model erdos_renyi --N 10 --num-graphs 10 --weighted --p-max 3`
- [ ] Generate comparison plots: success_prob vs gap alongside approx_ratio vs gap
- [ ] Scale up to N=12, N=14 on stronger machine
- [ ] Consider adding `filter_qaoa_monotonic.py` support for `success_prob` columns (currently only filters `approx_ratio`)

### Dependencies
- `networkx` must be installed (`pip install networkx`) — already in requirements.txt but was not installed on this machine's default Python
