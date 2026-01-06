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

# Import standalone plotting modules
from plot_p_sweep_ratio_vs_gap import plot_ratio_vs_gap
from plot_p_star_vs_gap import plot_p_star_vs_gap

# =========================================================================
# CONFIGURATION DEFAULTS
# =========================================================================

# Input/Output
DEFAULT_INPUT = 'outputs/weighted_gap_analysis_N12.csv'
DEFAULT_OUTPUT_DIR = 'outputs/pipeline_full_N12_weighted/'

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
    fqm.process_file(input_csv, verbose=True)
    
    # Determine output filename (process_file creates it automatically)
    input_path = Path(input_csv)
    filtered_filename = f"{input_path.stem}_filtered{input_path.suffix}"
    
    # If input is in output_dir, filtered file is there too
    # Otherwise it's in the same directory as input
    filtered_path = input_path.parent / filtered_filename
    
    return str(filtered_path)


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
    
    # Steps 3 & 4: Generate plots using standalone modules
    ratio_plot = None
    p_star_plot = None
    
    if not args.skip_plots:
        print("\n" + "=" * 70)
        print("  STEP 3: PLOTTING APPROXIMATION RATIO vs SPECTRAL GAP")
        print("=" * 70)
        ratio_plot = plot_ratio_vs_gap(filtered_output, args.output_dir)
        
        print("\n" + "=" * 70)
        print("  STEP 4: PLOTTING p* (MINIMUM DEPTH) vs SPECTRAL GAP")
        print("=" * 70)
        p_star_plot = plot_p_star_vs_gap(filtered_output, args.output_dir, DEFAULT_THRESHOLDS)
    
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
