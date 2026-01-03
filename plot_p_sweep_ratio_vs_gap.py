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

Usage:
    python plot_p_sweep_ratio_vs_gap.py                      # Use default input
    python plot_p_sweep_ratio_vs_gap.py path/to/data.csv     # Custom input
    
Can also be imported and called programmatically:
    from plot_p_sweep_ratio_vs_gap import plot_ratio_vs_gap
    output_path = plot_ratio_vs_gap('data.csv', 'output_dir/')
=========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import re
import os

# Configuration defaults
DEFAULT_INPUT = 'DataOutputs/QAOA_p_sweep_N12_p1to10.csv'
DEFAULT_OUTPUT_DIR = 'outputs/'


def plot_ratio_vs_gap(input_csv: str, output_dir: str = None, show_plot: bool = False) -> str:
    """
    Generate approximation ratio vs spectral gap plots for each QAOA depth.
    
    Args:
        input_csv: Path to QAOA results CSV file
        output_dir: Output directory for PNG file (default: same as input file)
        show_plot: Whether to display the plot interactively (default: False)
    
    Returns:
        Path to saved PNG file
    """
    print("=" * 70)
    print("  QAOA P-SWEEP: APPROXIMATION RATIO vs SPECTRAL GAP")
    print("=" * 70)
    
    # Load data
    print(f"\n📖 Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   Found {len(df)} graphs, N={df['N'].iloc[0]}")
    
    # Extract N for output filename
    N_value = df['N'].iloc[0]
    
    # Auto-detect available p values from column names
    ratio_cols = [col for col in df.columns if col.endswith('_approx_ratio')]
    
    if not ratio_cols:
        print(f"\n❌ ERROR: No QAOA approximation ratio columns found in {input_csv}")
        print(f"   Expected columns like 'p1_approx_ratio', 'p2_approx_ratio', etc.")
        print(f"   Available columns: {', '.join(df.columns.tolist())}")
        return None
    
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
            
            # Title with correlation and p-value
            title_color = 'green' if abs(corr) > 0.5 else 'orange' if abs(corr) > 0.3 else 'black'
            ax.set_title(f'p={p}: r={corr:+.3f}, p-val={stat_pval:.3f}', 
                         fontsize=11, fontweight='bold', color=title_color)
            
            # Print correlation
            significance = '***' if stat_pval < 0.001 else '**' if stat_pval < 0.01 else '*' if stat_pval < 0.05 else ''
            print(f"   p={p:2d}: r={corr:+.4f}, p-value={stat_pval:.4f} {significance}")
        else:
            ax.set_title(f'p={p}: No variance', fontsize=11)
            print(f"   p={p:2d}: No variance in data")
        
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
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_csv) or DEFAULT_OUTPUT_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input filename
    input_basename = os.path.basename(input_csv)  # Get filename without path
    input_name_no_ext = os.path.splitext(input_basename)[0]  # Remove extension
    
    # Replace "QAOA_p_sweep" with "ratio_vs_gap", or prepend if not present
    if input_name_no_ext.startswith("QAOA_p_sweep"):
        output_name = input_name_no_ext.replace("QAOA_p_sweep", "ratio_vs_gap", 1)
    else:
        output_name = f"ratio_vs_gap_{input_name_no_ext}"
    
    output_file = os.path.join(output_dir, f"{output_name}.png")
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
                print(f"   p={p:2d}: mean={mean_ratio:.4f}±{std_ratio:.4f}, "
                      f"range=[{min_ratio:.4f}, {max_ratio:.4f}]")
    
    print("\n" + "=" * 70)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_file


# =========================================================================
# COMMAND LINE INTERFACE
# =========================================================================

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_INPUT
    
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"❌ ERROR: Input file not found: {input_csv}")
        print(f"\n   This script requires QAOA p-sweep output files.")
        print(f"   Try one of these files instead:")
        print(f"     - DataOutputs/QAOA_p_sweep_N12_p1to10.csv")
        print(f"     - DataOutputs/QAOA_p_sweep_N12_p1to10_deg_2_only.csv")
        sys.exit(1)
    
    # Run the plotting function with interactive display
    plot_ratio_vs_gap(input_csv, output_dir=DEFAULT_OUTPUT_DIR, show_plot=True)
