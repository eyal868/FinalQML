#!/usr/bin/env python3
"""
=========================================================================
QAOA p-Sweep: Approximation Ratio vs Spectral Gap
=========================================================================
Visualize how the correlation between approximation ratio and spectral gap
changes across different QAOA depths (p=1 to p=10).

For each p value, show:
- Scatter plot: Δ_min vs approximation ratio
- Trend line
- Correlation coefficient and p-value
=========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import sys

# Configuration
DEFAULT_INPUT = 'outputs/QAOA_p_sweep_N10_p1to10.csv'
OUTPUT_DIR = 'outputs/'
P_VALUES = list(range(1, 11))  # p=1 to p=10

# Parse command line arguments
if len(sys.argv) > 1:
    INPUT_CSV = sys.argv[1]
else:
    INPUT_CSV = DEFAULT_INPUT

print("=" * 70)
print("  QAOA P-SWEEP: APPROXIMATION RATIO vs SPECTRAL GAP")
print("=" * 70)

# Load data
print(f"\n📖 Loading data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"   Found {len(df)} graphs, N={df['N'].iloc[0]}")

# Extract N for output filename
N_value = df['N'].iloc[0]

# Create figure with 2×5 subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(f'Approximation Ratio vs Spectral Gap Across QAOA Depths (N={N_value})', 
             fontsize=16, fontweight='bold', y=0.98)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

print(f"\n📊 Correlation Analysis by p:")

# Plot for each p value
for idx, p in enumerate(P_VALUES):
    ax = axes_flat[idx]
    
    # Get data for this p
    ratio_col = f'p{p}_approx_ratio'
    x = df['Delta_min'].values
    y = df[ratio_col].values
    
    # Remove any invalid values
    valid_mask = (y >= 0) & np.isfinite(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # Scatter plot
    ax.scatter(x_valid, y_valid, s=80, alpha=0.6, c='steelblue', 
               edgecolors='black', linewidth=0.5)
    
    # Calculate correlation
    if len(x_valid) > 1 and np.std(y_valid) > 0:
        corr, pval = pearsonr(x_valid, y_valid)
        
        # Add trend line
        z = np.polyfit(x_valid, y_valid, 1)
        p_fit = np.poly1d(z)
        x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
        ax.plot(x_trend, p_fit(x_trend), "r--", alpha=0.7, linewidth=2)
        
        # Title with correlation
        title_color = 'green' if abs(corr) > 0.5 else 'orange' if abs(corr) > 0.3 else 'black'
        ax.set_title(f'p={p}: r={corr:+.3f} (p={pval:.3f})', 
                     fontsize=11, fontweight='bold', color=title_color)
        
        # Print correlation
        significance = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"   p={p:2d}: r={corr:+.4f}, p-value={pval:.4f} {significance}")
    else:
        ax.set_title(f'p={p}: No variance', fontsize=11)
        print(f"   p={p:2d}: No variance in data")
    
    # Axis labels
    if idx >= 5:  # Bottom row
        ax.set_xlabel('Spectral Gap (Δ_min)', fontsize=10)
    if idx % 5 == 0:  # Left column
        ax.set_ylabel('Approximation Ratio', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Y-axis limits (reasonable range)
    y_min = max(0.5, y_valid.min() - 0.05) if len(y_valid) > 0 else 0.5
    y_max = min(1.05, y_valid.max() + 0.05) if len(y_valid) > 0 else 1.0
    ax.set_ylim([y_min, y_max])

plt.tight_layout()

# Save figure
output_file = f"{OUTPUT_DIR}p_sweep_ratio_vs_gap_N{N_value}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved: {output_file}")

# Summary statistics
print(f"\n📈 Summary Statistics:")
for p in P_VALUES:
    ratio_col = f'p{p}_approx_ratio'
    valid_ratios = df[df[ratio_col] >= 0][ratio_col]
    if len(valid_ratios) > 0:
        mean_ratio = valid_ratios.mean()
        std_ratio = valid_ratios.std()
        min_ratio = valid_ratios.min()
        max_ratio = valid_ratios.max()
        print(f"   p={p:2d}: mean={mean_ratio:.4f}±{std_ratio:.4f}, "
              f"range=[{min_ratio:.4f}, {max_ratio:.4f}]")

print("\n" + "=" * 70)

plt.show()

