#!/usr/bin/env python3
"""
=========================================================================
Plot Correlation: Spectral Gap vs QAOA Performance
=========================================================================
Visualize the relationship between minimum spectral gap (Δ_min) and
QAOA performance metrics.

Metrics analyzed:
1. Approximation ratio vs Δ_min
2. Optimizer iterations vs Δ_min  
3. Optimization time vs Δ_min
4. Final cost vs Δ_min
=========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Configuration
INPUT_CSV = 'outputs/QAOA_results_N10_p1.csv'
OUTPUT_DIR = 'outputs/'

# =========================================================================
# LOAD DATA
# =========================================================================

print("=" * 70)
print("  QAOA vs SPECTRAL GAP CORRELATION ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)
print(f"\n📖 Loaded {len(df)} graph results from {INPUT_CSV}")
print(f"   N = {df['N'].iloc[0]}, p = {df['p_layers'].iloc[0]}")

# Filter successful runs
df_success = df[df['Approximation_ratio'] >= 0]
print(f"   Successful runs: {len(df_success)}/{len(df)}")

# =========================================================================
# CORRELATION ANALYSIS
# =========================================================================

print(f"\n📊 Summary Statistics:")
print(f"   Δ_min: {df_success['Delta_min'].mean():.4f} ± {df_success['Delta_min'].std():.4f}")
print(f"   Approximation ratio: {df_success['Approximation_ratio'].mean():.4f} ± {df_success['Approximation_ratio'].std():.4f}")
print(f"   Optimizer iterations: {df_success['Optimizer_iterations'].mean():.1f} ± {df_success['Optimizer_iterations'].std():.1f}")
print(f"   Optimization time: {df_success['Optimization_time'].mean():.3f}s ± {df_success['Optimization_time'].std():.3f}s")

print(f"\n🔬 Correlation Analysis (Pearson):")

metrics = [
    ('Approximation_ratio', 'Approximation Ratio'),
    ('Optimizer_iterations', 'Optimizer Iterations'),
    ('Optimization_time', 'Optimization Time (s)'),
    ('Final_cost', 'Final Cost')
]

correlations = {}
for col, label in metrics:
    if df_success[col].std() > 0:  # Check for variance
        corr, p_value = pearsonr(df_success['Delta_min'], df_success[col])
        correlations[col] = (corr, p_value)
        print(f"   Δ_min vs {label:25s}: r = {corr:+.4f}, p = {p_value:.4f}")
    else:
        correlations[col] = (np.nan, np.nan)
        print(f"   Δ_min vs {label:25s}: No variance (all values identical)")

# =========================================================================
# VISUALIZATION
# =========================================================================

print(f"\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'QAOA Performance vs Spectral Gap (N={df["N"].iloc[0]}, p={df["p_layers"].iloc[0]})', 
             fontsize=16, fontweight='bold')

# Plot 1: Approximation Ratio vs Δ_min
ax = axes[0, 0]
ax.scatter(df_success['Delta_min'], df_success['Approximation_ratio'], 
           s=100, alpha=0.7, c=df_success['Optimizer_iterations'], 
           cmap='viridis', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Minimum Spectral Gap (Δ_min)', fontsize=12)
ax.set_ylabel('Approximation Ratio', fontsize=12)
ax.set_title('Approximation Ratio vs Spectral Gap', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.95, 1.05])  # Zoom in if all are near 1.0

if not np.isnan(correlations['Approximation_ratio'][0]):
    ax.text(0.05, 0.95, f"r = {correlations['Approximation_ratio'][0]:+.3f}\np = {correlations['Approximation_ratio'][1]:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax.text(0.05, 0.95, "No variance\n(all optimal)",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# Plot 2: Optimizer Iterations vs Δ_min
ax = axes[0, 1]
ax.scatter(df_success['Delta_min'], df_success['Optimizer_iterations'],
           s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Minimum Spectral Gap (Δ_min)', fontsize=12)
ax.set_ylabel('Optimizer Iterations', fontsize=12)
ax.set_title('Convergence Speed vs Spectral Gap', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

if not np.isnan(correlations['Optimizer_iterations'][0]):
    # Add trend line
    z = np.polyfit(df_success['Delta_min'], df_success['Optimizer_iterations'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_success['Delta_min'].min(), df_success['Delta_min'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    
    ax.text(0.05, 0.95, f"r = {correlations['Optimizer_iterations'][0]:+.3f}\np = {correlations['Optimizer_iterations'][1]:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper right')

# Plot 3: Optimization Time vs Δ_min
ax = axes[1, 0]
ax.scatter(df_success['Delta_min'], df_success['Optimization_time'],
           s=100, alpha=0.7, c='coral', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Minimum Spectral Gap (Δ_min)', fontsize=12)
ax.set_ylabel('Optimization Time (s)', fontsize=12)
ax.set_title('Optimization Time vs Spectral Gap', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

if not np.isnan(correlations['Optimization_time'][0]):
    ax.text(0.05, 0.95, f"r = {correlations['Optimization_time'][0]:+.3f}\np = {correlations['Optimization_time'][1]:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Final Cost vs Δ_min
ax = axes[1, 1]
ax.scatter(df_success['Delta_min'], df_success['Final_cost'],
           s=100, alpha=0.7, c='mediumseagreen', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Minimum Spectral Gap (Δ_min)', fontsize=12)
ax.set_ylabel('Final Cost (Energy)', fontsize=12)
ax.set_title('Final Cost vs Spectral Gap', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

if not np.isnan(correlations['Final_cost'][0]):
    ax.text(0.05, 0.95, f"r = {correlations['Final_cost'][0]:+.3f}\np = {correlations['Final_cost'][1]:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_file = f"{OUTPUT_DIR}QAOA_correlation_N{df['N'].iloc[0]}_p{df['p_layers'].iloc[0]}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_file}")

# =========================================================================
# ADDITIONAL ANALYSIS: HARDEST vs EASIEST
# =========================================================================

print(f"\n🔍 Detailed Comparison:")

if df_success['Approximation_ratio'].std() > 0:
    hardest_idx = df_success['Approximation_ratio'].idxmin()
    easiest_idx = df_success['Approximation_ratio'].idxmax()
    
    print(f"\n   Hardest graph (worst approximation ratio):")
    hardest = df_success.loc[hardest_idx]
    print(f"      Graph ID: {hardest['Graph_ID']}")
    print(f"      Δ_min: {hardest['Delta_min']:.6f}")
    print(f"      Approx ratio: {hardest['Approximation_ratio']:.6f}")
    print(f"      Iterations: {hardest['Optimizer_iterations']}")
    
    print(f"\n   Easiest graph (best approximation ratio):")
    easiest = df_success.loc[easiest_idx]
    print(f"      Graph ID: {easiest['Graph_ID']}")
    print(f"      Δ_min: {easiest['Delta_min']:.6f}")
    print(f"      Approx ratio: {easiest['Approximation_ratio']:.6f}")
    print(f"      Iterations: {easiest['Optimizer_iterations']}")
else:
    print(f"\n   ⚠️  All graphs achieved optimal solution (ratio = 1.0)")
    print(f"   → Cannot identify 'hardest' vs 'easiest' by approximation ratio")
    
    # Use iterations as proxy for difficulty
    hardest_idx = df_success['Optimizer_iterations'].idxmax()
    easiest_idx = df_success['Optimizer_iterations'].idxmin()
    
    print(f"\n   Slowest convergence (most iterations):")
    hardest = df_success.loc[hardest_idx]
    print(f"      Graph ID: {hardest['Graph_ID']}")
    print(f"      Δ_min: {hardest['Delta_min']:.6f}")
    print(f"      Iterations: {hardest['Optimizer_iterations']}")
    
    print(f"\n   Fastest convergence (fewest iterations):")
    easiest = df_success.loc[easiest_idx]
    print(f"      Graph ID: {easiest['Graph_ID']}")
    print(f"      Δ_min: {easiest['Delta_min']:.6f}")
    print(f"      Iterations: {easiest['Optimizer_iterations']}")

# =========================================================================
# INTERPRETATION
# =========================================================================

print(f"\n" + "=" * 70)
print(f"📋 INTERPRETATION:")
print(f"=" * 70)

if df_success['Approximation_ratio'].std() == 0 and df_success['Approximation_ratio'].mean() == 1.0:
    print(f"\n✅ All graphs solved to optimality (approx ratio = 1.0)!")
    print(f"   → N=10, p=1 QAOA is sufficient for all these 3-regular graphs")
    print(f"\n💡 Suggestions for future experiments:")
    print(f"   1. Test on larger N (12, 14) for harder instances")
    print(f"   2. Look at convergence metrics (iterations, time) as difficulty proxies")
    print(f"   3. Use less regular graphs (Erdős-Rényi, random geometric)")
    print(f"   4. Reduce optimizer max_iter to create 'partial convergence' scenarios")
    
    # Check if iterations correlate with Δ_min
    if not np.isnan(correlations['Optimizer_iterations'][0]):
        r_iter = correlations['Optimizer_iterations'][0]
        p_iter = correlations['Optimizer_iterations'][1]
        
        if abs(r_iter) > 0.3 and p_iter < 0.1:
            print(f"\n🎯 Interesting finding:")
            if r_iter > 0:
                print(f"   Graphs with larger Δ_min required MORE iterations (r={r_iter:.3f})")
            else:
                print(f"   Graphs with larger Δ_min required FEWER iterations (r={r_iter:.3f})")
            print(f"   This could indicate Δ_min affects convergence speed!")
        else:
            print(f"\n📊 No strong correlation between Δ_min and iterations (r={r_iter:.3f})")
else:
    print(f"\n📊 Some graphs not solved to optimality")
    r_approx = correlations['Approximation_ratio'][0]
    p_approx = correlations['Approximation_ratio'][1]
    
    if not np.isnan(r_approx) and p_approx < 0.05:
        print(f"   Significant correlation: r = {r_approx:.3f}, p = {p_approx:.4f}")
        if r_approx < 0:
            print(f"   → Smaller Δ_min correlates with worse QAOA performance!")
        else:
            print(f"   → Larger Δ_min correlates with worse QAOA performance!")
    else:
        print(f"   No significant correlation found")

print(f"\n" + "=" * 70)
print(f"✅ Analysis complete! Plot saved to {output_file}")
print(f"=" * 70)

plt.show()

