#!/usr/bin/env python3
"""
Tests for the monotonic filter logic in filter_qaoa_monotonic.py.

Verifies:
- P-value detection from column names
- Monotonicity filter correctness
- Edge cases (all valid, all invalid, NaN handling)
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filter_qaoa_monotonic import detect_p_values, apply_monotonic_filter


# =========================================================================
# TESTS: P-value detection
# =========================================================================

class TestDetectPValues:
    """Tests for detect_p_values column name parser."""

    def test_standard_columns(self):
        df = pd.DataFrame({
            'Graph_ID': [1],
            'p1_approx_ratio': [0.7],
            'p2_approx_ratio': [0.8],
            'p3_approx_ratio': [0.85],
        })
        assert detect_p_values(df) == [1, 2, 3]

    def test_non_contiguous(self):
        df = pd.DataFrame({
            'p1_approx_ratio': [0.7],
            'p5_approx_ratio': [0.8],
            'p10_approx_ratio': [0.9],
        })
        assert detect_p_values(df) == [1, 5, 10]

    def test_no_ratio_columns(self):
        df = pd.DataFrame({'Graph_ID': [1], 'Delta_min': [0.5]})
        assert detect_p_values(df) == []


# =========================================================================
# TESTS: Monotonic filter
# =========================================================================

class TestMonotonicFilter:
    """Tests for apply_monotonic_filter correctness."""

    def _make_df(self, ratios_list, p_values=None):
        """Helper: create a DataFrame from a list of ratio sequences.
        Each element is a list of ratios for p=1,2,...,len."""
        if p_values is None:
            p_values = list(range(1, len(ratios_list[0]) + 1))
        data = {'Graph_ID': list(range(len(ratios_list)))}
        for i, p in enumerate(p_values):
            data[f'p{p}_approx_ratio'] = [row[i] for row in ratios_list]
        return pd.DataFrame(data), p_values

    def test_already_monotonic(self):
        """Monotonically increasing ratios should not be modified."""
        df, p_vals = self._make_df([[0.7, 0.8, 0.85, 0.9]])
        df_filtered, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        assert stats['total_invalidated'] == 0
        assert stats['graphs_affected'] == 0
        # Values unchanged
        for p in p_vals:
            assert df_filtered.iloc[0][f'p{p}_approx_ratio'] == df.iloc[0][f'p{p}_approx_ratio']

    def test_single_violation(self):
        """One drop: [0.7, 0.8, 0.75, 0.9] -> p3 should be NaN."""
        df, p_vals = self._make_df([[0.7, 0.8, 0.75, 0.9]])
        df_filtered, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        assert stats['total_invalidated'] == 1
        assert pd.isna(df_filtered.iloc[0]['p3_approx_ratio'])
        # p4 (0.9) is still valid because 0.9 > 0.8 (last valid)
        assert df_filtered.iloc[0]['p4_approx_ratio'] == 0.9

    def test_cascading_violations(self):
        """Ratios that drop and stay below: [0.9, 0.7, 0.6, 0.5].
        p2 drops below p1 (0.7 < 0.9) -> NaN
        p3 drops below last valid (0.6 < 0.9) -> NaN
        p4 drops below last valid (0.5 < 0.9) -> NaN"""
        df, p_vals = self._make_df([[0.9, 0.7, 0.6, 0.5]])
        df_filtered, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        assert stats['total_invalidated'] == 3
        assert not pd.isna(df_filtered.iloc[0]['p1_approx_ratio'])
        assert pd.isna(df_filtered.iloc[0]['p2_approx_ratio'])
        assert pd.isna(df_filtered.iloc[0]['p3_approx_ratio'])
        assert pd.isna(df_filtered.iloc[0]['p4_approx_ratio'])

    def test_recovery_after_drop(self):
        """[0.7, 0.8, 0.75, 0.85] -> p3 NaN, p4 valid (0.85 > 0.8)."""
        df, p_vals = self._make_df([[0.7, 0.8, 0.75, 0.85]])
        df_filtered, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        assert stats['total_invalidated'] == 1
        assert pd.isna(df_filtered.iloc[0]['p3_approx_ratio'])
        assert df_filtered.iloc[0]['p4_approx_ratio'] == 0.85

    def test_multiple_graphs(self):
        """Multiple graphs: one clean, one with violations."""
        df, p_vals = self._make_df([
            [0.7, 0.8, 0.85],   # clean
            [0.7, 0.6, 0.85],   # p2 violates
        ])
        df_filtered, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        assert stats['total_invalidated'] == 1
        assert stats['graphs_affected'] == 1
        # Graph 0 untouched
        assert df_filtered.iloc[0]['p2_approx_ratio'] == 0.8
        # Graph 1: p2 invalidated
        assert pd.isna(df_filtered.iloc[1]['p2_approx_ratio'])

    def test_flat_sequence_is_valid(self):
        """Equal consecutive values (ratio(p) == ratio(p-1)) should be kept."""
        df, p_vals = self._make_df([[0.7, 0.7, 0.7]])
        df_filtered, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        assert stats['total_invalidated'] == 0

    def test_existing_nan_handled(self):
        """Pre-existing NaN values should be skipped, not counted as violations."""
        df, p_vals = self._make_df([[0.7, float('nan'), 0.8]])
        df_filtered, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        # p3=0.8 > p1=0.7 (last valid), so p3 should stay
        assert stats['total_invalidated'] == 0
        assert df_filtered.iloc[0]['p3_approx_ratio'] == 0.8

    def test_stats_totals(self):
        """Verify stats dictionary has correct total counts."""
        df, p_vals = self._make_df([
            [0.7, 0.8, 0.75],
            [0.7, 0.6, 0.5],
        ])
        _, stats = apply_monotonic_filter(df.copy(), p_vals, verbose=False)
        assert stats['total_values'] == 6  # 2 graphs * 3 p-values
        assert stats['total_graphs'] == 2
        assert stats['total_invalidated'] == 3  # 1 from graph0, 2 from graph1
        assert stats['graphs_affected'] == 2
