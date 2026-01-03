#!/usr/bin/env python3
"""
=========================================================================
QAOA Analysis Pipeline: Complete Workflow Runner
=========================================================================
Orchestrates the full QAOA analysis workflow:

1. QAOA p-sweep analysis on graphs (qaoa_analysis.py)
2. Monotonic filtering to remove optimization artifacts (filter_qaoa_monotonic.py)
3. Approximation ratio vs spectral gap plots (plot_p_sweep_ratio_vs_gap.py)
4. Minimum depth (p*) vs spectral gap plots (plot_p_star_vs_gap.py)

Usage:
    python run_qaoa_pipeline.py                          # Use defaults
    python run_qaoa_pipeline.py --input path/to/file.csv # Custom input
    python run_qaoa_pipeline.py --skip-qaoa              # Skip QAOA (use existing)
    python run_qaoa_pipeline.py --help                   # Show all options
=========================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

# =========================================================================
# CONFIGURATION DEFAULTS
# =========================================================================

# Input/Output
DEFAULT_INPUT = 'outputs/Delta_min_3_regular_N12_sparse_k2_final.csv'
DEFAULT_OUTPUT_DIR = 'outputs/pipeline_full_N14/'

# QAOA Parameters
DEFAULT_DEGENERACY = None  # None = process all, int = filter to specific degeneracy
DEFAULT_P_VALUES = list(range(1, 11))  # Test p=1,2,...,10
DEFAULT_MAX_ITER = 500
DEFAULT_NUM_SHOTS = 10000

# p* Analysis Thresholds
DEFAULT_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]

# Parallel Processing
DEFAULT_PARALLEL = True          # Enable parallel by default
DEFAULT_WORKERS = None           # None = use all CPU cores

# =========================================================================
# ARGUMENT PARSER
# =========================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='QAOA Analysis Pipeline: Complete workflow runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_qaoa_pipeline.py
  python run_qaoa_pipeline.py --input outputs/Delta_min_N10.csv
  python run_qaoa_pipeline.py --degeneracy 4 --skip-qaoa
  python run_qaoa_pipeline.py --p-max 5 --output-dir results/
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT,
                        help=f'Input spectral gap CSV file (default: {DEFAULT_INPUT})')
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for all results (default: {DEFAULT_OUTPUT_DIR})')
    
    # QAOA Parameters
    parser.add_argument('--degeneracy', '-d', type=int, default=DEFAULT_DEGENERACY,
                        help='Filter graphs by degeneracy (default: process all)')
    parser.add_argument('--p-min', type=int, default=1,
                        help='Minimum p value to test (default: 1)')
    parser.add_argument('--p-max', type=int, default=10,
                        help='Maximum p value to test (default: 10)')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER,
                        help=f'Max optimizer iterations (default: {DEFAULT_MAX_ITER})')
    parser.add_argument('--shots', type=int, default=DEFAULT_NUM_SHOTS,
                        help=f'Number of measurement shots (default: {DEFAULT_NUM_SHOTS})')
    
    # Skip Options
    parser.add_argument('--skip-qaoa', action='store_true',
                        help='Skip QAOA analysis (use existing results file)')
    parser.add_argument('--skip-filter', action='store_true',
                        help='Skip monotonic filtering')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    
    # QAOA Results File (when skipping QAOA)
    parser.add_argument('--qaoa-results', type=str, default=None,
                        help='Existing QAOA results file (required with --skip-qaoa)')
    
    # Parallel Processing (use dest='parallel' so both flags control the same attribute)
    parser.add_argument('--parallel', dest='parallel', action='store_true',
                        help='Enable parallel processing')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                        help='Disable parallel processing (run sequentially)')
    parser.set_defaults(parallel=DEFAULT_PARALLEL)
    parser.add_argument('--workers', '-w', type=int, default=DEFAULT_WORKERS,
                        help='Number of worker processes (default: all CPU cores)')
    
    return parser.parse_args()


# =========================================================================
# PIPELINE STEPS
# =========================================================================

def run_qaoa_analysis(input_csv: str, output_dir: str, degeneracy: int,
                      p_values: list, max_iter: int, num_shots: int,
                      use_parallel: bool = True, num_workers: int = None) -> str:
    """
    Step 1: Run QAOA p-sweep analysis on graphs.
    
    Returns: Path to output CSV file
    """
    print("\n" + "=" * 70)
    print("  STEP 1: QAOA P-SWEEP ANALYSIS")
    print("=" * 70)
    
    # Import the analysis module
    import qaoa_analysis as qa
    
    # Override module configuration
    qa.INPUT_CSV = input_csv
    qa.FILTER_DEGENERACY = degeneracy
    qa.P_VALUES_TO_TEST = p_values
    qa.MAX_OPTIMIZER_ITERATIONS = max_iter
    qa.NUM_SHOTS = num_shots
    qa.USE_PARALLEL = use_parallel
    qa.NUM_WORKERS = num_workers
    
    # Generate output filename
    input_path = Path(input_csv)
    input_stem = input_path.stem
    
    # Extract N from input filename or data
    import pandas as pd
    df_temp = pd.read_csv(input_csv)
    N_value = df_temp['N'].iloc[0]
    
    # Build output filename
    deg_suffix = f"_deg{degeneracy}" if degeneracy else ""
    p_range = f"_p{min(p_values)}to{max(p_values)}"
    output_filename = f"QAOA_p_sweep_N{N_value}{deg_suffix}{p_range}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Override output filename in module
    qa.OUTPUT_FILENAME = output_path
    
    # Run analysis
    qa.analyze_graphs_from_csv(input_csv)
    
    return output_path


def run_monotonic_filter(input_csv: str, output_dir: str) -> str:
    """
    Step 2: Apply monotonic filter to remove optimization artifacts.
    
    Returns: Path to filtered output CSV file
    """
    print("\n" + "=" * 70)
    print("  STEP 2: MONOTONIC FILTERING")
    print("=" * 70)
    
    # Import the filter module
    import filter_qaoa_monotonic as fqm
    
    # Process the file
    stats = fqm.process_file(input_csv, verbose=True)
    
    # Determine output filename (process_file creates it automatically)
    input_path = Path(input_csv)
    filtered_filename = f"{input_path.stem}_filtered{input_path.suffix}"
    
    # If input is in output_dir, filtered file is there too
    # Otherwise it's in the same directory as input
    filtered_path = input_path.parent / filtered_filename
    
    return str(filtered_path)


def run_ratio_vs_gap_plot(input_csv: str, output_dir: str) -> str:
    """
    Step 3: Generate approximation ratio vs spectral gap plots.
    
    Returns: Path to output PNG file
    """
    print("\n" + "=" * 70)
    print("  STEP 3: PLOTTING APPROXIMATION RATIO vs SPECTRAL GAP")
    print("=" * 70)
    
    # Import required modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr
    import re
    
    # Load data
    print(f"\n📖 Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   Found {len(df)} graphs, N={df['N'].iloc[0]}")
    
    N_value = df['N'].iloc[0]
    
    # Auto-detect p values
    ratio_cols = [col for col in df.columns if col.endswith('_approx_ratio')]
    if not ratio_cols:
        print(f"❌ ERROR: No approx_ratio columns found in {input_csv}")
        return None
    
    P_VALUES = sorted([int(re.search(r'p(\d+)_approx_ratio', col).group(1)) 
                       for col in ratio_cols])
    max_p = max(P_VALUES)
    min_p = min(P_VALUES)
    
    print(f"   Detected p values: {min_p} to {max_p}")
    
    # Create subplot grid
    n_plots = len(P_VALUES)
    n_cols = min(5, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.suptitle(f'Approximation Ratio vs Spectral Gap (N={N_value})', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    # Handle axes flattening
    if n_plots == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes_flat = axes.flatten()
    
    print(f"\n📊 Correlation Analysis by p:")
    
    # Plot each p value
    for idx, p in enumerate(P_VALUES):
        ax = axes_flat[idx]
        ratio_col = f'p{p}_approx_ratio'
        x = df['Delta_min'].values
        y = df[ratio_col].values
        
        # Remove invalid values (NaN from filtering)
        valid_mask = np.isfinite(y) & (y >= 0)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Scatter plot
        ax.scatter(x_valid, y_valid, s=80, alpha=0.6, c='steelblue',
                   edgecolors='black', linewidth=0.5)
        
        # Correlation and trend line
        if len(x_valid) > 1 and np.std(y_valid) > 0:
            corr, stat_pval = pearsonr(x_valid, y_valid)
            z = np.polyfit(x_valid, y_valid, 1)
            p_fit = np.poly1d(z)
            x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
            ax.plot(x_trend, p_fit(x_trend), "r--", alpha=0.7, linewidth=2)
            
            title_color = 'green' if abs(corr) > 0.5 else 'orange' if abs(corr) > 0.3 else 'black'
            ax.set_title(f'p={p}: r={corr:+.3f}', fontsize=11, fontweight='bold', color=title_color)
            
            sig = '***' if stat_pval < 0.001 else '**' if stat_pval < 0.01 else '*' if stat_pval < 0.05 else ''
            print(f"   p={p:2d}: r={corr:+.4f}, p-value={stat_pval:.4f} {sig}")
        else:
            ax.set_title(f'p={p}: N/A', fontsize=11)
        
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Spectral Gap (Δ_min)', fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel('Approximation Ratio', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if len(y_valid) > 0:
            ax.set_ylim([max(0.5, y_valid.min() - 0.05), min(1.05, y_valid.max() + 0.05)])
    
    # Hide unused subplots
    for idx in range(len(P_VALUES), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    input_stem = Path(input_csv).stem
    if input_stem.startswith("QAOA_p_sweep"):
        output_name = input_stem.replace("QAOA_p_sweep", "ratio_vs_gap", 1)
    else:
        output_name = f"ratio_vs_gap_{input_stem}"
    
    output_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure saved: {output_path}")
    return output_path


def run_p_star_plot(input_csv: str, output_dir: str, thresholds: list) -> str:
    """
    Step 4: Generate p* (minimum depth required) vs spectral gap plots.
    
    Returns: Path to output PNG file
    """
    print("\n" + "=" * 70)
    print("  STEP 4: PLOTTING p* (MINIMUM DEPTH) vs SPECTRAL GAP")
    print("=" * 70)
    
    # Import required modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    import re
    
    # Load data
    print(f"\n📖 Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   Found {len(df)} graphs, N={df['N'].iloc[0]}")
    
    N_value = df['N'].iloc[0]
    
    # Auto-detect p values
    ratio_cols = [col for col in df.columns if col.endswith('_approx_ratio')]
    if not ratio_cols:
        print(f"❌ ERROR: No approx_ratio columns found in {input_csv}")
        return None
    
    P_VALUES = sorted([int(re.search(r'p(\d+)_approx_ratio', col).group(1)) 
                       for col in ratio_cols])
    max_p = max(P_VALUES)
    
    # Calculate p* for each threshold
    print(f"\n🎯 Calculating p* for thresholds: {thresholds}")
    
    p_star_data = {}
    for threshold in thresholds:
        p_star_list = []
        for idx, row in df.iterrows():
            for p in P_VALUES:
                ratio = row[f'p{p}_approx_ratio']
                if pd.notna(ratio) and ratio >= threshold:
                    p_star_list.append(p)
                    break
            else:
                p_star_list.append(max_p + 1)
        
        p_star_data[threshold] = p_star_list
        reached = sum(1 for ps in p_star_list if ps <= max_p)
        mean_p = np.mean([ps for ps in p_star_list if ps <= max_p]) if reached > 0 else np.nan
        print(f"   θ={threshold:.2f}: {reached}/{len(df)} graphs reach it" + 
              (f", mean p*={mean_p:.2f}" if reached > 0 else ""))
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['#e74c3c', '#9b59b6', '#f39c12', '#2ecc71', '#3498db']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, threshold in enumerate(thresholds):
        x = df['Delta_min'].values
        y = np.array(p_star_data[threshold])
        
        reached_mask = y <= max_p
        
        if np.any(reached_mask):
            ax.scatter(x[reached_mask], y[reached_mask],
                      s=100, alpha=0.7, c=colors[i % len(colors)], 
                      marker=markers[i % len(markers)],
                      edgecolors='black', linewidth=1,
                      label=f'θ={threshold:.2f}')
            
            if np.sum(reached_mask) > 2:
                x_r, y_r = x[reached_mask], y[reached_mask]
                z = np.polyfit(x_r, y_r, 1)
                p_fit = np.poly1d(z)
                x_trend = np.linspace(x_r.min(), x_r.max(), 100)
                ax.plot(x_trend, p_fit(x_trend), '--', color=colors[i % len(colors)],
                       alpha=0.5, linewidth=2)
        
        if np.any(~reached_mask):
            ax.scatter(x[~reached_mask], y[~reached_mask],
                      s=100, alpha=0.4, c=colors[i % len(colors)], marker='x',
                      linewidth=2)
    
    ax.scatter([], [], s=100, alpha=0.4, c='gray', marker='x', linewidth=2, label='Not Reached')
    
    ax.set_xlabel('Spectral Gap (Δ_min)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Minimum p Required (p*)', fontsize=14, fontweight='bold')
    ax.set_title(f'Minimum QAOA Depth vs Spectral Gap (N={N_value})',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.set_ylim([0.5, max_p + 1.5])
    ax.set_yticks(range(1, max_p + 2))
    ax.set_yticklabels([str(i) if i <= max_p else 'N/R' for i in range(1, max_p + 2)])
    
    plt.tight_layout()
    
    # Save figure
    input_stem = Path(input_csv).stem
    if input_stem.startswith("QAOA_p_sweep"):
        output_name = input_stem.replace("QAOA_p_sweep", "p_star_vs_gap", 1)
    else:
        output_name = f"p_star_vs_gap_{input_stem}"
    
    output_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure saved: {output_path}")
    
    # Correlation analysis
    print(f"\n🔬 Correlation Analysis (p* vs Δ_min):")
    for threshold in thresholds:
        y = np.array(p_star_data[threshold])
        reached_mask = y <= max_p
        if np.sum(reached_mask) > 2:
            x_valid = df['Delta_min'].values[reached_mask]
            y_valid = y[reached_mask]
            r, pval = pearsonr(x_valid, y_valid)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f"   θ={threshold:.2f}: r={r:+.4f}, p-value={pval:.4f} {sig}")
    
    return output_path


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def main():
    """Main pipeline execution."""
    from multiprocessing import cpu_count
    
    args = parse_args()
    
    # Parallel mode: --parallel enables, --no-parallel disables, default from DEFAULT_PARALLEL
    use_parallel = args.parallel
    num_workers = args.workers if args.workers else cpu_count()
    
    print("=" * 70)
    print("  QAOA ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Input:        {args.input}")
    print(f"   Output dir:   {args.output_dir}")
    print(f"   Degeneracy:   {args.degeneracy if args.degeneracy else 'All'}")
    print(f"   P values:     {args.p_min} to {args.p_max}")
    print(f"   Parallel:     {'Enabled (' + str(num_workers) + ' workers)' if use_parallel else 'Disabled'}")
    print(f"   Skip QAOA:    {args.skip_qaoa}")
    print(f"   Skip Filter:  {args.skip_filter}")
    print(f"   Skip Plots:   {args.skip_plots}")
    
    # Validate input
    if not os.path.exists(args.input) and not args.skip_qaoa:
        print(f"\n❌ ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {args.output_dir}")
    
    # Track timing
    start_time = time.time()
    
    # Determine QAOA results file
    if args.skip_qaoa:
        if args.qaoa_results:
            qaoa_output = args.qaoa_results
        else:
            print("\n❌ ERROR: --skip-qaoa requires --qaoa-results to specify existing file")
            sys.exit(1)
        print(f"\n⏭️  Skipping QAOA analysis, using: {qaoa_output}")
    else:
        # Step 1: Run QAOA analysis
        p_values = list(range(args.p_min, args.p_max + 1))
        qaoa_output = run_qaoa_analysis(
            input_csv=args.input,
            output_dir=args.output_dir,
            degeneracy=args.degeneracy,
            p_values=p_values,
            max_iter=args.max_iter,
            num_shots=args.shots,
            use_parallel=use_parallel,
            num_workers=num_workers
        )
    
    # Step 2: Monotonic filtering
    if args.skip_filter:
        filtered_output = qaoa_output
        print(f"\n⏭️  Skipping monotonic filter, using: {filtered_output}")
    else:
        filtered_output = run_monotonic_filter(qaoa_output, args.output_dir)
    
    # Steps 3 & 4: Generate plots
    if not args.skip_plots:
        ratio_plot = run_ratio_vs_gap_plot(filtered_output, args.output_dir)
        p_star_plot = run_p_star_plot(filtered_output, args.output_dir, DEFAULT_THRESHOLDS)
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n📊 Output files:")
    print(f"   QAOA Results:    {qaoa_output}")
    if not args.skip_filter:
        print(f"   Filtered:        {filtered_output}")
    if not args.skip_plots:
        print(f"   Ratio Plot:      {ratio_plot}")
        print(f"   p* Plot:         {p_star_plot}")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()

