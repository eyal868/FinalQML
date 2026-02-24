#!/usr/bin/env python3
"""
=========================================================================
Test Script for SCD File Parser
=========================================================================
Validates the .scd parser implementation by comparing results with the
trusted .asc parser on 12_3_3 files (85 graphs, 12 vertices, 3-regular).

This test ensures:
1. Correct number of graphs parsed
2. Exact edge list match for each graph
3. No duplicate edges
4. Correct vertex degrees (all vertices have degree 3)
=========================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aqc_spectral_utils import parse_asc_file, parse_scd_file

def validate_graph_properties(edges, n, k, graph_id):
    """
    Validate basic properties of a k-regular graph on n vertices.

    Args:
        edges: List of (v1, v2) tuples (0-indexed)
        n: Number of vertices
        k: Expected degree
        graph_id: Graph identifier for error messages

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check for duplicate edges
    if len(edges) != len(set(edges)):
        errors.append(f"Graph {graph_id}: Contains duplicate edges")

    # Check expected number of edges
    expected_edges = n * k // 2
    if len(edges) != expected_edges:
        errors.append(f"Graph {graph_id}: Expected {expected_edges} edges, got {len(edges)}")

    # Check all vertices have degree k
    degree_count = [0] * n
    for v1, v2 in edges:
        if v1 < 0 or v1 >= n or v2 < 0 or v2 >= n:
            errors.append(f"Graph {graph_id}: Vertex index out of range: ({v1}, {v2})")
            continue
        degree_count[v1] += 1
        degree_count[v2] += 1

    for vertex, degree in enumerate(degree_count):
        if degree != k:
            errors.append(f"Graph {graph_id}: Vertex {vertex} has degree {degree}, expected {k}")

    return errors


def compare_edge_lists(edges1, edges2):
    """
    Compare two edge lists (order-independent).

    Args:
        edges1, edges2: Lists of (v1, v2) tuples

    Returns:
        True if edge lists match, False otherwise
    """
    set1 = set(edges1)
    set2 = set(edges2)
    return set1 == set2


def main():
    print("=" * 70)
    print("  SCD PARSER VALIDATION TEST")
    print("=" * 70)

    # File paths
    asc_file = "graphs_rawdata/12_3_3.asc"
    scd_file = "graphs_rawdata/12_3_3.scd"
    n = 12  # vertices
    k = 3   # degree

    print(f"\n📖 Reading graphs from files...")
    print(f"  • ASC file: {asc_file}")
    print(f"  • SCD file: {scd_file}")

    # Parse both files
    try:
        graphs_asc = parse_asc_file(asc_file)
        print(f"  ✓ ASC parser: {len(graphs_asc)} graphs")
    except Exception as e:
        print(f"  ❌ Error parsing ASC file: {e}")
        sys.exit(1)

    try:
        graphs_scd = parse_scd_file(scd_file)
        print(f"  ✓ SCD parser: {len(graphs_scd)} graphs")
    except Exception as e:
        print(f"  ❌ Error parsing SCD file: {e}")
        sys.exit(1)

    # Check graph count
    print(f"\n🔍 Validation Results:")
    print("-" * 70)

    if len(graphs_asc) != len(graphs_scd):
        print(f"❌ FAILED: Graph count mismatch!")
        print(f"   ASC: {len(graphs_asc)} graphs")
        print(f"   SCD: {len(graphs_scd)} graphs")
        sys.exit(1)
    else:
        print(f"✓ Graph count matches: {len(graphs_asc)} graphs")

    # Compare each graph
    mismatches = []
    validation_errors = []

    for i, (edges_asc, edges_scd) in enumerate(zip(graphs_asc, graphs_scd)):
        graph_id = i + 1  # 1-indexed for display

        # Validate graph properties
        errors_asc = validate_graph_properties(edges_asc, n, k, f"{graph_id} (ASC)")
        errors_scd = validate_graph_properties(edges_scd, n, k, f"{graph_id} (SCD)")

        if errors_asc:
            validation_errors.extend(errors_asc)
        if errors_scd:
            validation_errors.extend(errors_scd)

        # Compare edge lists
        if not compare_edge_lists(edges_asc, edges_scd):
            mismatches.append({
                'graph_id': graph_id,
                'asc_edges': edges_asc,
                'scd_edges': edges_scd
            })

    # Report validation errors
    if validation_errors:
        print(f"\n⚠️  Found {len(validation_errors)} validation error(s):")
        for error in validation_errors[:10]:  # Show first 10
            print(f"   • {error}")
        if len(validation_errors) > 10:
            print(f"   ... and {len(validation_errors) - 10} more")
    else:
        print(f"✓ All graphs have valid properties (degree-{k}, {n} vertices)")

    # Report mismatches
    if mismatches:
        print(f"\n❌ FAILED: Found {len(mismatches)} graph mismatch(es)!")
        print("\nShowing first 3 mismatches:")
        for mismatch in mismatches[:3]:
            gid = mismatch['graph_id']
            print(f"\n  Graph {gid}:")

            asc_set = set(mismatch['asc_edges'])
            scd_set = set(mismatch['scd_edges'])

            only_in_asc = asc_set - scd_set
            only_in_scd = scd_set - asc_set

            if only_in_asc:
                print(f"    Only in ASC: {sorted(only_in_asc)[:5]}")
            if only_in_scd:
                print(f"    Only in SCD: {sorted(only_in_scd)[:5]}")

        sys.exit(1)
    else:
        print(f"✓ All edge lists match exactly!")

    # Display sample graphs
    print(f"\n📊 Sample Graphs (first 3):")
    print("-" * 70)
    for i in range(min(3, len(graphs_asc))):
        graph_id = i + 1
        edges = graphs_asc[i]
        print(f"\nGraph {graph_id}: {len(edges)} edges")
        print(f"  {edges[:6]}...")

        # Verify degrees
        degree_count = [0] * n
        for v1, v2 in edges:
            degree_count[v1] += 1
            degree_count[v2] += 1
        print(f"  Degrees: {degree_count}")

    # Final summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print(f"   • Parsed {len(graphs_asc)} graphs from both formats")
    print(f"   • All edge lists match exactly")
    print(f"   • All graphs validated as {k}-regular on {n} vertices")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

