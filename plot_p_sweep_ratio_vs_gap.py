#!/usr/bin/env python3
"""
=========================================================================
QAOA p-Sweep: Approximation Ratio vs Spectral Gap
=========================================================================
Visualize how the correlation between approximation ratio and spectral gap
changes across different QAOA depths.

Automatically detects available p values from the CSV file and creates
an appropriate subplot grid.

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
import re
import os

# Configuration
DEFAULT_INPUT = 'outputs/Delta_min_3_regular_N12_sparse_k2_final.csv'
OUTPUT_DIR = 'outputs/'

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

# Auto-detect available p values from column names
ratio_cols = [col for col in df.columns if col.endswith('_approx_ratio')]
P_VALUES = sorted([int(re.search(r'p(\d+)_approx_ratio', col).group(1)) 
                   for col in ratio_cols])
max_p = max(P_VALUES)
min_p = min(P_VALUES)

print(f"   Detected p values: {min_p} to {max_p} ({len(P_VALUES)} total)")

# Calculate optimal subplot grid (prefer wider than tall, max 5 columns)
n_plots = len(P_VALUES)
n_cols = min(5, n_plots)
n_rows = int(np.ceil(n_plots / n_cols))

print(f"   Creating {n_rows}×{n_cols} subplot grid")

# Create figure with dynamic subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
fig.suptitle(f'Approximation Ratio vs Spectral Gap Across QAOA Depths (N={N_value})', 
             fontsize=16, fontweight='bold', y=0.99)

# Flatten axes for easier iteration (handle both 1D and 2D cases)
if n_plots == 1:
    axes_flat = [axes]
elif n_rows == 1 or n_cols == 1:
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
else:
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
        corr, stat_pval = pearsonr(x_valid, y_valid)
        
        # Add trend line
        z = np.polyfit(x_valid, y_valid, 1)
        p_fit = np.poly1d(z)
        x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
        ax.plot(x_trend, p_fit(x_trend), "r--", alpha=0.7, linewidth=2)
        
        # Title with correlation
        title_color = 'green' if abs(corr) > 0.5 else 'orange' if abs(corr) > 0.3 else 'black'
        ax.set_title(f'layers({p}): r={corr:+.3f} (stat_p={stat_pval:.3f})', 
                     fontsize=11, fontweight='bold', color=title_color)
        
        # Print correlation
        significance = '***' if stat_pval < 0.001 else '**' if stat_pval < 0.01 else '*' if stat_pval < 0.05 else ''
        print(f"   layers({p:2d}): r={corr:+.4f}, stat_p-value={stat_pval:.4f} {significance}")
    else:
        ax.set_title(f'layers({p}): No variance', fontsize=11)
        print(f"   layers({p:2d}): No variance in data")
    
    # Axis labels (dynamic based on grid)
    if idx >= (n_rows - 1) * n_cols:  # Bottom row
        ax.set_xlabel('Spectral Gap (Δ_min)', fontsize=10)
    if idx % n_cols == 0:  # Left column
        ax.set_ylabel('Approximation Ratio', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Y-axis limits (reasonable range)
    y_min = max(0.5, y_valid.min() - 0.05) if len(y_valid) > 0 else 0.5
    y_max = min(1.05, y_valid.max() + 0.05) if len(y_valid) > 0 else 1.0
    ax.set_ylim([y_min, y_max])

# Hide any unused subplots
for idx in range(len(P_VALUES), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()

# Generate output filename based on input filename
input_basename = os.path.basename(INPUT_CSV)  # Get filename without path
input_name_no_ext = os.path.splitext(input_basename)[0]  # Remove extension

# Replace "QAOA_p_sweep" with "p_sweep_ratio_vs_gap", or prepend if not present
if input_name_no_ext.startswith("QAOA_p_sweep"):
    output_name = input_name_no_ext.replace("QAOA_p_sweep", "p_sweep_ratio_vs_gap", 1)
else:
    output_name = f"p_sweep_ratio_vs_gap_{input_name_no_ext}"

output_file = f"{OUTPUT_DIR}{output_name}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved: {output_file}")

# Summary statistics
print(f"\n📈 Summary Statistics:")
for p in P_VALUES:
    ratio_col = f'p{p}_approx_ratio'
    if ratio_col in df.columns:
        valid_ratios = df[df[ratio_col] >= 0][ratio_col]
        if len(valid_ratios) > 0:
            mean_ratio = valid_ratios.mean()
            std_ratio = valid_ratios.std()
            min_ratio = valid_ratios.min()
            max_ratio = valid_ratios.max()
            print(f"   layers({p:2d}): mean={mean_ratio:.4f}±{std_ratio:.4f}, "
                  f"range=[{min_ratio:.4f}, {max_ratio:.4f}]")

print("\n" + "=" * 70)

plt.show()

