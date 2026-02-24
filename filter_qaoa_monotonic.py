#!/usr/bin/env python3
"""
=========================================================================
QAOA Monotonic Filter: Remove Optimization Artifacts
=========================================================================
Apply strict monotonicity constraint to QAOA approximation ratios:
If ratio(p) < ratio(p-1), mark ratio(p) as invalid (NaN).

This removes optimization failures from correlation analysis while
preserving the theoretical expectation that QAOA at depth p should
achieve at least as well as depth p-1.

Usage:
    python filter_qaoa_monotonic.py                                    # Process all
    python filter_qaoa_monotonic.py outputs/qaoa_unweighted/N10/qaoa_sweep_N10_p1to10.csv  # Specific file
    python filter_qaoa_monotonic.py file1.csv file2.csv                # Multiple files
=========================================================================
"""

import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path
import glob

from qaoa_utils import detect_p_values

def apply_monotonic_filter(df, p_values, verbose=True):
    """
    Apply strict monotonicity filter: mark ratio(p) as NaN if ratio(p) < ratio(p-1).

    Returns:
        df: Modified dataframe
        stats: Dictionary with filtering statistics
    """
    total_invalidated = 0
    graphs_affected = 0
    invalidation_details = []

    for idx in range(len(df)):
        graph_id = df.loc[idx, 'Graph_ID']
        prev_ratio = 0
        graph_invalidations = []

        for p in p_values:
            ratio_col = f'p{p}_approx_ratio'
            current_ratio = df.loc[idx, ratio_col]

            # Check if current ratio is valid (not NaN) and violates monotonicity
            if pd.notna(current_ratio):
                if current_ratio < prev_ratio:
                    # Mark as invalid - optimization failure
                    graph_invalidations.append((p, current_ratio, prev_ratio))
                    df.loc[idx, ratio_col] = np.nan
                    total_invalidated += 1
                else:
                    # Update prev_ratio only if current is valid
                    prev_ratio = current_ratio

        if graph_invalidations:
            graphs_affected += 1
            invalidation_details.append((graph_id, graph_invalidations))

    stats = {
        'total_invalidated': total_invalidated,
        'total_values': len(df) * len(p_values),
        'graphs_affected': graphs_affected,
        'total_graphs': len(df),
        'invalidation_details': invalidation_details
    }

    return df, stats

def print_statistics(input_file, p_values, stats, verbose=True):
    """Print detailed filtering statistics."""
    print(f"\n📊 Processing: {input_file}")
    print(f"   Found {stats['total_graphs']} graphs with p values: {min(p_values)}-{max(p_values)}")

    if verbose and stats['invalidation_details']:
        print(f"\n   Detailed filtering results:")
        for graph_id, invalidations in stats['invalidation_details']:
            invalid_str = ', '.join([f"p={p}: {curr:.3f}<{prev:.3f}"
                                     for p, curr, prev in invalidations])
            print(f"   Graph {graph_id:2d}: {len(invalidations)} values invalidated ({invalid_str})")

    # Summary
    print(f"\n   Summary:")
    print(f"   - Total values invalidated: {stats['total_invalidated']}/{stats['total_values']} "
          f"({100*stats['total_invalidated']/stats['total_values']:.1f}%)")
    print(f"   - Graphs affected: {stats['graphs_affected']}/{stats['total_graphs']} "
          f"({100*stats['graphs_affected']/stats['total_graphs']:.1f}%)")

def process_file(input_file, verbose=True):
    """Process a single CSV file."""
    # Read data
    df = pd.read_csv(input_file)

    # Detect p values
    p_values = detect_p_values(df)

    # Apply filtering
    df_filtered, stats = apply_monotonic_filter(df, p_values, verbose=verbose)

    # Print statistics
    print_statistics(input_file, p_values, stats, verbose=verbose)

    # Save filtered data
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    df_filtered.to_csv(output_file, index=False)

    print(f"\n   ✅ Saved: {output_file}")

    return stats

def main():
    """Main entry point."""
    print("=" * 70)
    print("  QAOA MONOTONIC FILTER: Remove Optimization Artifacts")
    print("=" * 70)

    # Determine which files to process
    if len(sys.argv) > 1:
        # Files specified on command line
        input_files = sys.argv[1:]
    else:
        # Default: process all qaoa_sweep*.csv files in outputs/
        input_files = glob.glob('outputs/**/qaoa_sweep_*.csv', recursive=True) + glob.glob('outputs/**/weighted_qaoa_*.csv', recursive=True)
        # Exclude already filtered files
        input_files = [f for f in input_files if '_filtered' not in f]

        if not input_files:
            print("\n❌ No qaoa_sweep_*.csv or weighted_qaoa_*.csv files found in outputs/")
            print("   Usage: python filter_qaoa_monotonic.py [file1.csv] [file2.csv] ...")
            sys.exit(1)

    print(f"\n📁 Files to process: {len(input_files)}")
    for f in input_files:
        print(f"   - {f}")

    # Process each file
    all_stats = []
    for input_file in input_files:
        try:
            stats = process_file(input_file, verbose=True)
            all_stats.append((input_file, stats))
        except Exception as e:
            print(f"\n❌ Error processing {input_file}: {e}")
            continue

    # Overall summary
    if len(all_stats) > 1:
        print("\n" + "=" * 70)
        print("  OVERALL SUMMARY")
        print("=" * 70)

        total_invalidated = sum(s['total_invalidated'] for _, s in all_stats)
        total_values = sum(s['total_values'] for _, s in all_stats)
        total_graphs_affected = sum(s['graphs_affected'] for _, s in all_stats)
        total_graphs = sum(s['total_graphs'] for _, s in all_stats)

        print(f"\n   Across {len(all_stats)} files:")
        print(f"   - Total values invalidated: {total_invalidated}/{total_values} "
              f"({100*total_invalidated/total_values:.1f}%)")
        print(f"   - Graphs affected: {total_graphs_affected}/{total_graphs} "
              f"({100*total_graphs_affected/total_graphs:.1f}%)")

    print("\n" + "=" * 70)
    print("  ✅ Filtering complete!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   Run your plotting scripts with the filtered files:")
    for input_file, _ in all_stats:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
        print(f"   python plot_p_sweep_ratio_vs_gap.py {output_file}")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()


