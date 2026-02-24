#!/usr/bin/env python3
"""
=========================================================================
QAOA Shared Utilities
=========================================================================
Common helper functions used across multiple QAOA analysis and plotting
scripts. Centralizes duplicated logic for:
- P-value detection from CSV column names
- Spectral gap column auto-detection
- Graph loading from CSV files
- Weight string parsing
=========================================================================
"""

import re
import ast
import pandas as pd


def detect_p_values(df):
    """Auto-detect available p values from column names.

    Scans DataFrame columns for the pattern 'p{N}_approx_ratio' and
    returns the sorted list of integer p values found.

    Args:
        df: pandas DataFrame with QAOA result columns

    Returns:
        Sorted list of integer p values
    """
    ratio_cols = [col for col in df.columns if col.endswith('_approx_ratio')]
    p_values = sorted([int(re.search(r'p(\d+)_approx_ratio', col).group(1))
                       for col in ratio_cols])
    return p_values


def detect_gap_column(df):
    """Auto-detect the spectral gap column in a DataFrame.

    Supports both unweighted ('Delta_min') and weighted ('Weighted_Delta_min')
    result formats.

    Args:
        df: pandas DataFrame with spectral gap data

    Returns:
        (gap_col, gap_label) tuple, or (None, None) if no gap column found
    """
    if 'Delta_min' in df.columns:
        return 'Delta_min', 'Spectral Gap (Δ_min)'
    elif 'Weighted_Delta_min' in df.columns:
        return 'Weighted_Delta_min', 'Weighted Spectral Gap (Δ_min)'
    return None, None


def load_graph_from_csv(csv_filename, graph_id):
    """
    Load graph data from CSV file by graph ID.

    Args:
        csv_filename: Path to spectral gap CSV file
        graph_id: Integer graph ID to look up

    Returns:
        dict with keys: 'graph_id', 'N', 'edges', 'Delta_min',
        's_at_min', 'Max_degeneracy', 'Max_cut_value'
    """
    df = pd.read_csv(csv_filename)
    row = df[df['Graph_ID'] == graph_id]

    if len(row) == 0:
        raise ValueError(f"Graph ID {graph_id} not found in {csv_filename}")

    row = row.iloc[0]

    # Parse edges string (stored as Python list literal)
    edges = ast.literal_eval(row['Edges'])

    return {
        'graph_id': int(row['Graph_ID']),
        'N': int(row['N']),
        'edges': edges,
        'Delta_min': float(row['Delta_min']),
        's_at_min': float(row['s_at_min']),
        'Max_degeneracy': int(row['Max_degeneracy']),
        'Max_cut_value': int(row['Max_cut_value'])
    }


def parse_weights_string(weights_str: str) -> list:
    """
    Parse weights from CSV string representation.

    Handles format like: "[np.float64(1.23), np.float64(4.56), ...]"
    or simple list format: "[1.23, 4.56, ...]"

    Args:
        weights_str: String representation of weights list

    Returns:
        List of float weights
    """
    # Extract all numbers using regex for np.float64 format
    pattern = r'np\.float64\(([\d.e+-]+)\)'
    matches = re.findall(pattern, weights_str)

    if matches:
        return [float(x) for x in matches]

    # Fallback: try ast.literal_eval for simple lists
    try:
        return list(ast.literal_eval(weights_str))
    except Exception:
        raise ValueError(f"Could not parse weights: {weights_str[:100]}...")
