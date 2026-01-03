# Paper Context - Quick Reference

> **📄 Full Baseline Context**: See [`PROJECT_CONTEXT.md`](../.cursor/Context/PROJECT_CONTEXT.md) in the project root for comprehensive documentation.

This file provides a quick reference to the current paper and its relationship to the codebase.

---

## Paper Information

**Title**: "Numerical Analyses of The Relation Between QAOA Performance and The Minimum Spectral Gap"

**Authors**: Eyal Chai-Ezra

**Status**: Active research project - currently being expanded

**PDF Location**: `This_Project_Paper_QAOA_spectral_gap.pdf`

---

## Research Question

Does the QAOA circuit depth (p) depend on the minimum spectral gap (g_min) in the same way that Adiabatic Quantum Computing runtime scales as T ∝ 1/g_min²?

**Answer**: **No** - g_min is not a reliable predictor of QAOA performance.

---

## Key Counter-Intuitive Findings

1. **Negative Correlation**: Instances with smaller g_min (AQC-hard) achieve **higher** approximation ratios in QAOA
2. **Inverted Depth**: Instances with larger g_min require **more** layers to reach target accuracy
3. **Degeneracy Critical**: Controlling for ground state degeneracy is essential to see true relationship
4. **Optimization Confounds**: ~45% of data filtered due to classical optimizer failures

---

## Current Datasets

- **N=10**: 19 graphs, 3-regular (analyzed)
- **N=12**: 85 graphs, 3-regular (analyzed)
- **Circuit depths**: p = 1 to 10
- **Degeneracy subsets**: k=2 (31 graphs), k=4 (26 graphs)

---

## Code-to-Paper Mapping

### Main Analysis Scripts

| Paper Section | Script | Key Config |
|---------------|--------|------------|
| Section 3.1.2 (g_min calculation) | `spectral_gap_analysis.py` | M=200 (N=10), M=20 (N=12) |
| Section 3.2 (QAOA analysis) | `qaoa_analysis.py` | p=1-10, COBYLA, 500 iters |
| Section 4.1 (filtering) | `filter_qaoa_monotonic.py` | Monotonicity filter |
| Section 4.1 (depth sweep) | `plot_p_sweep_ratio_vs_gap.py` | Multi-panel correlations |
| Section 4.2 (critical depth) | `plot_p_star_vs_gap.py` | Thresholds 0.75-0.95 |

### Figure Generation

| Figure in Paper | Script | Output File |
|-----------------|--------|-------------|
| Figure: Energy Spectrum Graph 18 | `plot_example_spectrum.py` | `example_full_spectrum_N12_graph18.pdf` |
| Figure: Gap vs Degeneracy | `plot_delta_vs_degeneracy.py` | `Delta_min_3_regular_N12_res20_delta_vs_degeneracy.pdf` |
| Figure: Main Result (k=2) | `plot_p_sweep_ratio_vs_gap.py` | `p_sweep_ratio_vs_gap_N12_p1to10_deg_2_only_filtered.png` |
| Figure: p* Analysis | `plot_p_star_vs_gap.py` | `p_star_vs_gap_N12_p1to10_deg_2_only_filtered.png` |
| Figure: Optimization Failures | `plot_optimization_failure_examples.py` | `optimization_failure_examples.png` |

---

## Expansion Directions (Section 6)

### From Paper's Future Work

1. **Generalizability**: Extend to Vertex Cover, SAT, Number Partitioning
2. **Precision**: Adaptive/binary search for g_min calculation
3. **Optimization**: Warm-start, multi-start, basin-hopping methods

### Technical Extensions

- **Larger N**: 14, 16, 18+ (computational challenge)
- **More graphs**: 4-regular, 5-regular (data exists, not yet analyzed)
- **Higher p**: >10 layers (optimization difficulty increases)
- **Noise models**: NISQ-relevant analysis

---

## Critical Development Rules

### ⚠️ Always Remember

1. **Control for degeneracy** - analyze k=2, k=4 separately
2. **Apply monotonicity filter** - remove optimization artifacts
3. **Document all parameters** - in CSV filenames and metadata
4. **Generate both formats** - PNG for paper, keep CSV source
5. **This is active research** - we're expanding, not just maintaining

### Data Quality Checklist

- [ ] Degeneracy controlled
- [ ] Filtered and unfiltered versions
- [ ] Complete metadata (g_min, s_min, degeneracy, max_cut)
- [ ] Reproducible (fixed seed, documented config)
- [ ] Publication-quality figures (300 DPI, statistics shown)

---

## Quick Links

- **Full Context**: [`PROJECT_CONTEXT.md`](../.cursor/Context/PROJECT_CONTEXT.md)
- **User Guide**: [`README.md`](../README.md)
- **Optimization Deep Dive**: [`QAOA_OPTIMIZATION_CHALLENGES.md`](../QAOA_OPTIMIZATION_CHALLENGES.md)
- **Repository**: https://github.com/eyal868/FinalQML

---

**Last Updated**: November 25, 2025




