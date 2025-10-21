#!/usr/bin/env python3
"""
=========================================================================
QAOA p*: Minimum Depth Required vs Spectral Gap
=========================================================================
Analyze the relationship between spectral gap (Δ_min) and the minimum
QAOA depth (p*) required to reach different approximation ratio thresholds.

For each threshold (0.75, 0.80, 0.85, 0.90, 0.95), calculate p* for each
graph and plot p* vs Δ_min to test the hypothesis:
"Graphs with smaller spectral gaps require larger p to achieve good performance"
=========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import sys
import re
import os

# Configuration
DEFAULT_INPUT = 'outputs/QAOA_p_sweep_N12_p1to10_deg_2_only_filtered.csv'
OUTPUT_DIR = 'outputs/'
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]

# Parse command line arguments
if len(sys.argv) > 1:
    INPUT_CSV = sys.argv[1]
else:
    INPUT_CSV = DEFAULT_INPUT

print("=" * 70)
print("  QAOA p*: MINIMUM DEPTH REQUIRED vs SPECTRAL GAP")
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

# Calculate p* for each threshold and each graph
print(f"\n🎯 Calculating p* for each threshold...")

p_star_data = {}
for threshold in THRESHOLDS:
    p_star_list = []
    for idx, row in df.iterrows():
        # Find minimum p that achieves threshold
        for p in P_VALUES:
            ratio_col = f'p{p}_approx_ratio'
            if row[ratio_col] >= threshold:
                p_star_list.append(p)
                break
        else:
            # Threshold not reached by max_p
            p_star_list.append(max_p + 1)
    
    p_star_data[threshold] = p_star_list
    
    # Calculate statistics
    reached = sum(1 for p in p_star_list if p <= max_p)
    mean_p = np.mean([p for p in p_star_list if p <= max_p]) if reached > 0 else np.nan
    
    print(f"   Threshold {threshold:.2f}: {reached}/{len(df)} graphs reach it, "
          f"mean p*={mean_p:.2f}" if reached > 0 else f"mean p*=N/A")

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Color palette for different thresholds
colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']
markers = ['o', 's', '^', 'D', 'v']

# Plot p* vs Δ_min for each threshold
for i, threshold in enumerate(THRESHOLDS):
    x = df['Delta_min'].values
    y = np.array(p_star_data[threshold])
    
    # Separate reached and not-reached
    reached_mask = y <= max_p
    
    # Plot reached points
    if np.any(reached_mask):
        ax.scatter(x[reached_mask], y[reached_mask], 
                  s=100, alpha=0.7, c=colors[i], marker=markers[i],
                  edgecolors='black', linewidth=1, 
                  label=f'Threshold {threshold:.2f}')
        
        # Add trend line if enough points
        if np.sum(reached_mask) > 2:
            x_reached = x[reached_mask]
            y_reached = y[reached_mask]
            z = np.polyfit(x_reached, y_reached, 1)
            p_fit = np.poly1d(z)
            x_trend = np.linspace(x_reached.min(), x_reached.max(), 100)
            ax.plot(x_trend, p_fit(x_trend), '--', color=colors[i], 
                   alpha=0.5, linewidth=2)
    
    # Plot not-reached points (p*=max_p+1) with different marker
    if np.any(~reached_mask):
        ax.scatter(x[~reached_mask], y[~reached_mask], 
                  s=100, alpha=0.4, c=colors[i], marker='x',
                  edgecolors='black', linewidth=2)

# Formatting
ax.set_xlabel('Spectral Gap (Δ_min)', fontsize=14, fontweight='bold')
ax.set_ylabel('Minimum p Required (p*)', fontsize=14, fontweight='bold')
ax.set_title(f'Minimum QAOA Depth vs Spectral Gap (N={N_value})', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.9)

# Y-axis: integer values from 1 to max_p+1
ax.set_ylim([0.5, max_p + 1.5])
ax.set_yticks(range(1, max_p + 2))
ax.set_yticklabels([str(i) if i <= max_p else 'N/R' for i in range(1, max_p + 2)])

# Add horizontal line at p=max_p
ax.axhline(y=max_p + 0.5, color='red', linestyle=':', linewidth=1, alpha=0.3)
ax.text(ax.get_xlim()[1], max_p + 0.5, ' Not Reached', 
        verticalalignment='center', fontsize=9, color='red', alpha=0.7)

plt.tight_layout()

# Generate output filename based on input filename
input_basename = os.path.basename(INPUT_CSV)  # Get filename without path
input_name_no_ext = os.path.splitext(input_basename)[0]  # Remove extension

# Replace "QAOA_p_sweep" with "p_star_vs_gap", or prepend if not present
if input_name_no_ext.startswith("QAOA_p_sweep"):
    output_name = input_name_no_ext.replace("QAOA_p_sweep", "p_star_vs_gap", 1)
else:
    output_name = f"p_star_vs_gap_{input_name_no_ext}"

output_file = f"{OUTPUT_DIR}{output_name}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved: {output_file}")

# Correlation analysis
print(f"\n🔬 Correlation Analysis (p* vs Δ_min):")
print("   " + "-" * 60)
print("   Threshold | Pearson r | p-value | Spearman ρ | p-value |")
print("   " + "-" * 60)

for threshold in THRESHOLDS:
    y = np.array(p_star_data[threshold])
    
    # Only use graphs that reached the threshold
    reached_mask = y <= max_p
    
    if np.sum(reached_mask) > 2:
        x_valid = df['Delta_min'].values[reached_mask]
        y_valid = y[reached_mask]
        
        # Pearson correlation
        r_pearson, p_pearson = pearsonr(x_valid, y_valid)
        
        # Spearman correlation (rank-based, more robust)
        r_spearman, p_spearman = spearmanr(x_valid, y_valid)
        
        sig_pearson = '***' if p_pearson < 0.001 else '**' if p_pearson < 0.01 else '*' if p_pearson < 0.05 else ''
        sig_spearman = '***' if p_spearman < 0.001 else '**' if p_spearman < 0.01 else '*' if p_spearman < 0.05 else ''
        
        print(f"   {threshold:.2f}     | {r_pearson:+.4f}   | {p_pearson:.4f}  | "
              f"{r_spearman:+.4f}    | {p_spearman:.4f}  | {sig_spearman}")
    else:
        print(f"   {threshold:.2f}     | N/A (too few points)")

print("   " + "-" * 60)
print("   Significance: *** p<0.001, ** p<0.01, * p<0.05")

# Interpretation
print(f"\n💡 Interpretation:")
print("   Negative correlation → Smaller Δ_min requires larger p*")
print("   Positive correlation → Larger Δ_min requires larger p*")
print("   |r| > 0.5 = Strong, |r| > 0.3 = Moderate, |r| < 0.3 = Weak")

# Additional statistics
print(f"\n📊 p* Distribution by Threshold:")
for threshold in THRESHOLDS:
    p_star = np.array(p_star_data[threshold])
    p_star_valid = p_star[p_star <= max_p]
    
    if len(p_star_valid) > 0:
        print(f"   {threshold:.2f}: min={p_star_valid.min()}, "
              f"median={np.median(p_star_valid):.0f}, "
              f"mean={p_star_valid.mean():.2f}, "
              f"max={p_star_valid.max()}, "
              f"std={p_star_valid.std():.2f}")
    else:
        print(f"   {threshold:.2f}: No graphs reached this threshold")

print("\n" + "=" * 70)

plt.show()

