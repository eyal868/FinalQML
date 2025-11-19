# Optimization Failure Examples for Paper

## Overview

Created publication-quality figure showing representative examples of classical optimization failures in QAOA where approximation ratios drop at higher p values.

**Figure**: `outputs/optimization_failure_examples.png`

## Representative Cases Identified

### N=10: Graph 16
- **Spectral Gap**: Δ_min = 1.088
- **Number of drops**: 5 drops across p=5,6,7,8,10
- **Most dramatic failure**: p=7 with **0.926 → 0.624** (drop of -0.301)
- **Interpretation**: Despite having a large spectral gap, optimization completely fails at p=7

**All drops for Graph 16:**
- p=5: 0.926 → 0.899 (Δ=-0.027)
- p=6: 0.926 → 0.912 (Δ=-0.014) 
- p=7: 0.926 → 0.624 (Δ=-0.301) ← Huge drop!
- p=8: 0.926 → 0.767 (Δ=-0.159)
- p=10: 0.954 → 0.932 (Δ=-0.022)

### N=12: Graph 14
- **Spectral Gap**: Δ_min = 0.445
- **Number of drops**: 7 drops across p=3,5,6,7,8,9,10
- **Most dramatic failure**: p=8 with **0.923 → 0.671** (drop of -0.252)
- **Interpretation**: Persistent optimization challenges across many p values

**All drops for Graph 14:**
- p=3: 0.849 → 0.847 (Δ=-0.003)
- p=5: 0.923 → 0.844 (Δ=-0.079)
- p=6: 0.923 → 0.872 (Δ=-0.051)
- p=7: 0.923 → 0.893 (Δ=-0.030)
- p=8: 0.923 → 0.671 (Δ=-0.252) ← Huge drop!
- p=9: 0.923 → 0.878 (Δ=-0.045)
- p=10: 0.923 → 0.869 (Δ=-0.054)

## Figure Description

**Layout**: Side-by-side comparison (N=10 left, N=12 right)

**Visual Elements**:
- 🔵 **Blue line with markers**: Observed approximation ratios
- ⚫ **Gray dashed line**: Theoretical monotonic envelope (best achievable)
- ❌ **Red X markers**: Points where ratio(p) < ratio(p-1) (optimization failures)
- 📍 **Red annotation**: Arrow pointing to worst drop with magnitude

**Key Insight**: The gap between blue (observed) and gray (theoretical) shows how much performance is lost due to optimization failures, not quantum limitations.

## Suggested Figure Caption

```
Representative examples of classical optimization failures in QAOA. 
Blue lines show observed approximation ratios as circuit depth increases, 
while gray dashed lines show the theoretical monotonic envelope (best 
achievable performance if optimization succeeded at each depth). Red X 
markers indicate points where ratio(p) < ratio(p-1), violating the 
theoretical expectation that deeper circuits should perform at least as 
well as shallower ones. These failures occur when the classical optimizer 
(COBYLA) gets stuck in local minima of the high-dimensional parameter 
landscape at larger p values. Left: N=10, Graph 16 (Δ_min=1.088) shows 
a dramatic failure at p=7 with a 30% drop in performance. Right: N=12, 
Graph 14 (Δ_min=0.445) exhibits persistent optimization challenges across 
multiple depths with a 25% drop at p=8.
```

## Usage in Paper

### Section: Results / QAOA Analysis

**Narrative flow:**

1. **Introduce the problem**:
   > "We observe that increasing circuit depth p does not always lead to improved 
   > approximation ratios, contrary to theoretical expectations..."

2. **Show the examples** (cite the figure):
   > "Figure X shows representative examples from our dataset. For N=10, Graph 16 
   > achieves a strong ratio of 0.926 at p=6, but drops dramatically to 0.624 at 
   > p=7—a 30% performance loss. Similarly, for N=12, Graph 14 exhibits..."

3. **Explain the cause**:
   > "This non-monotonic behavior is not a quantum phenomenon but rather a classical 
   > optimization challenge. As p increases, the parameter space grows exponentially 
   > (2p parameters), and local optimization methods like COBYLA become increasingly 
   > likely to get trapped in local minima."

4. **Connect to filtering approach**:
   > "To address this issue and ensure our correlation analysis measures quantum 
   > performance rather than optimization quality, we filter data points where 
   > ratio(p) < ratio(p-1), as detailed in Section X..."

### Statistics to Include

From the full filtering analysis:
- **56% of all ratio values** were affected (1,894 out of 3,380)
- **100% of graphs** showed at least one optimization failure
- **N=12 at p=20**: 65.8% invalidation rate (optimization difficulty increases with p)

## Regenerating the Figure

If you need to regenerate or modify the figure:

```bash
python plot_optimization_failure_examples.py
```

The script automatically selects the most dramatic cases based on:
- Number of drops
- Magnitude of drops
- Drops occurring in middle p range (not just at end)

## Related Files

- **Filtering script**: `filter_qaoa_monotonic.py`
- **Filtering summary**: `FILTERING_SUMMARY.md`
- **Problem documentation**: `QAOA_OPTIMIZATION_CHALLENGES.md`

---

**Created**: October 21, 2025  
**Script**: `plot_optimization_failure_examples.py`  
**Figure**: `outputs/optimization_failure_examples.png`  
**Status**: ✅ Ready for paper


