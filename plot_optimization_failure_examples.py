#!/usr/bin/env python3
"""
=========================================================================
QAOA Optimization Failure Examples: Publication Figure
=========================================================================
Identify and visualize representative cases of classical optimization
failures in QAOA, where approximation ratios drop at higher p values
despite the theoretical expectation of monotonic improvement.

This creates a publication-quality figure showing:
- Actual approximation ratios vs theoretical monotonic envelope
- Clear examples of optimization getting stuck in local minima
- Representative cases from N=10 and N=12

Output: outputs/optimization_failure_examples.png
=========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Configuration
N10_INPUT = 'outputs/QAOA_p_sweep_N10_p1to10.csv'
N12_INPUT = 'outputs/QAOA_p_sweep_N12_p1to10.csv'
OUTPUT_FILE = 'DataOutputs/optimization_failure_examples.png'

def detect_p_values(df):
    """Auto-detect available p values from column names."""
    ratio_cols = [col for col in df.columns if col.endswith('_approx_ratio')]
    p_values = sorted([int(re.search(r'p(\d+)_approx_ratio', col).group(1)) 
                       for col in ratio_cols])
    return p_values

def analyze_graph_drops(df, graph_idx, p_values):
    """
    Analyze a single graph for optimization failures.
    
    Returns:
        drops: list of (p, ratio_before, ratio_after, magnitude) tuples
        ratios: array of all ratios
        monotonic_envelope: array of running maximum (theoretical best)
    """
    ratios = []
    for p in p_values:
        ratio_col = f'p{p}_approx_ratio'
        ratios.append(df.loc[graph_idx, ratio_col])
    
    ratios = np.array(ratios)
    
    # Compute monotonic envelope (running maximum)
    monotonic_envelope = np.maximum.accumulate(ratios)
    
    # Detect drops
    drops = []
    prev_ratio = 0
    for i, p in enumerate(p_values):
        current_ratio = ratios[i]
        if current_ratio < prev_ratio:
            magnitude = current_ratio - prev_ratio
            drops.append((p, prev_ratio, current_ratio, magnitude))
        else:
            prev_ratio = current_ratio
    
    return drops, ratios, monotonic_envelope

def score_graph(df, graph_idx, p_values):
    """
    Score a graph for how representative it is as an optimization failure example.
    
    Higher score = better example (more drops, larger magnitude, varied positions)
    """
    drops, ratios, _ = analyze_graph_drops(df, graph_idx, p_values)
    
    if len(drops) == 0:
        return 0, None
    
    # Scoring factors
    num_drops = len(drops)
    avg_magnitude = np.mean([abs(d[3]) for d in drops])
    max_magnitude = max([abs(d[3]) for d in drops])
    
    # Bonus for drops in middle range (not just at end)
    middle_drops = sum(1 for d in drops if 3 <= d[0] <= 8)
    middle_bonus = middle_drops * 0.5
    
    # Combined score
    score = (num_drops * 2.0 +  # Number of drops
             avg_magnitude * 10.0 +  # Average drop size
             max_magnitude * 15.0 +  # Worst drop
             middle_bonus)  # Drops in middle p range
    
    info = {
        'num_drops': num_drops,
        'avg_magnitude': avg_magnitude,
        'max_magnitude': max_magnitude,
        'worst_drop': max(drops, key=lambda d: abs(d[3]))
    }
    
    return score, info

def find_best_example(df, p_values, N):
    """Find the most representative optimization failure example."""
    print(f"\n🔍 Analyzing {len(df)} graphs for N={N}...")
    
    best_score = 0
    best_idx = None
    best_info = None
    
    for idx in range(len(df)):
        score, info = score_graph(df, idx, p_values)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_info = info
    
    if best_idx is None:
        print(f"   ❌ No graphs with drops found!")
        return None, None, None
    
    graph_id = df.loc[best_idx, 'Graph_ID']
    delta_min = df.loc[best_idx, 'Delta_min']
    
    print(f"\n   ✅ Best candidate: Graph {graph_id}")
    print(f"      Δ_min = {delta_min:.3f}")
    print(f"      {best_info['num_drops']} drops detected")
    p_worst, before, after, mag = best_info['worst_drop']
    print(f"      Worst drop: p={p_worst} ({before:.3f} → {after:.3f}, Δ={mag:.3f})")
    
    return best_idx, graph_id, delta_min

def plot_example(ax, df, graph_idx, p_values, N, graph_id, delta_min):
    """Plot a single optimization failure example."""
    drops, ratios, monotonic_envelope = analyze_graph_drops(df, graph_idx, p_values)
    
    # Plot actual ratios (blue line with markers)
    ax.plot(p_values, ratios, 'o-', color='steelblue', linewidth=2.5, 
            markersize=8, label='Observed ratio', zorder=3)
    
    # Plot monotonic envelope (gray dashed line)
    ax.plot(p_values, monotonic_envelope, '--', color='gray', linewidth=2, 
            alpha=0.7, label='Theoretical (monotonic)', zorder=2)
    
    # Mark drops with red X
    drop_p_values = [d[0] for d in drops]
    drop_ratios = [d[2] for d in drops]  # ratio_after
    ax.scatter(drop_p_values, drop_ratios, marker='x', s=200, 
              color='red', linewidths=3, label='Optimization failure', zorder=4)
    
    # Annotate worst drop
    if drops:
        worst_drop = max(drops, key=lambda d: abs(d[3]))
        p_worst, before, after, mag = worst_drop
        
        # Arrow pointing to the drop
        ax.annotate(f'Drop: {mag:.3f}',
                   xy=(p_worst, after), xytext=(p_worst + 1.5, after + 0.08),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='red', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Circuit Depth (p)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Approximation Ratio', fontsize=13, fontweight='bold')
    ax.set_title(f'N={N}, Graph #{graph_id}, Δ_min={delta_min:.3f}', 
                fontsize=12, pad=10)
    
    ax.set_xticks(p_values)
    ax.set_ylim([0.5, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Add subtle background highlighting for drop regions
    for drop in drops:
        p_drop = drop[0]
        ax.axvspan(p_drop - 0.3, p_drop + 0.3, alpha=0.05, color='red', zorder=1)

def main():
    """Main execution."""
    print("=" * 70)
    print("  QAOA OPTIMIZATION FAILURE EXAMPLES")
    print("=" * 70)
    
    # Load data
    print(f"\n📖 Loading data...")
    df_n10 = pd.read_csv(N10_INPUT)
    df_n12 = pd.read_csv(N12_INPUT)
    print(f"   N=10: {len(df_n10)} graphs")
    print(f"   N=12: {len(df_n12)} graphs")
    
    # Detect p values
    p_values_n10 = detect_p_values(df_n10)
    p_values_n12 = detect_p_values(df_n12)
    
    print(f"\n   p values: {min(p_values_n10)}-{max(p_values_n10)}")
    
    # Find best examples
    print("\n" + "=" * 70)
    print("  FINDING REPRESENTATIVE CASES")
    print("=" * 70)
    
    idx_n10, graph_id_n10, delta_n10 = find_best_example(df_n10, p_values_n10, 10)
    idx_n12, graph_id_n12, delta_n12 = find_best_example(df_n12, p_values_n12, 12)
    
    if idx_n10 is None or idx_n12 is None:
        print("\n❌ Could not find suitable examples!")
        return
    
    # Create figure
    print("\n" + "=" * 70)
    print("  CREATING FIGURE")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Representative Examples of Classical Optimization Failures in QAOA', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot examples
    plot_example(ax1, df_n10, idx_n10, p_values_n10, 10, graph_id_n10, delta_n10)
    plot_example(ax2, df_n12, idx_n12, p_values_n12, 12, graph_id_n12, delta_n12)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"\nN=10 Example: Graph {graph_id_n10} (Δ_min={delta_n10:.3f})")
    drops_n10, _, _ = analyze_graph_drops(df_n10, idx_n10, p_values_n10)
    print(f"  Drops at p = {[d[0] for d in drops_n10]}")
    for p, before, after, mag in drops_n10[:3]:  # Show first 3
        print(f"    p={p}: {before:.3f} → {after:.3f} (Δ={mag:.3f})")
    
    print(f"\nN=12 Example: Graph {graph_id_n12} (Δ_min={delta_n12:.3f})")
    drops_n12, _, _ = analyze_graph_drops(df_n12, idx_n12, p_values_n12)
    print(f"  Drops at p = {[d[0] for d in drops_n12]}")
    for p, before, after, mag in drops_n12[:3]:  # Show first 3
        print(f"    p={p}: {before:.3f} → {after:.3f} (Δ={mag:.3f})")
    
    print("\n" + "=" * 70)
    print("  💡 SUGGESTED FIGURE CAPTION")
    print("=" * 70)
    print("""
Representative examples of classical optimization failures in QAOA. 
Blue lines show observed approximation ratios as circuit depth increases, 
while gray dashed lines show the theoretical monotonic envelope (best 
achievable performance if optimization succeeded at each depth). Red X 
markers indicate points where ratio(p) < ratio(p-1), violating the 
theoretical expectation that deeper circuits should perform at least as 
well as shallower ones. These failures occur when the classical optimizer 
(COBYLA) gets stuck in local minima of the high-dimensional parameter 
landscape at larger p values.
    """.strip())
    
    print("\n" + "=" * 70)
    
    plt.show()

if __name__ == '__main__':
    main()


