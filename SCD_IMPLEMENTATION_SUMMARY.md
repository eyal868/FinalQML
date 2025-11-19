# SCD File Parser Implementation Summary

## Overview

Successfully implemented support for reading GENREG `.scd` (shortcode) binary graph files, complementing the existing `.asc` text file parser.

## Implementation Details

### 1. Added to `aqc_spectral_utils.py`

**New Graph I/O Section:**

- `extract_graph_params(filename)` - Extracts n, k, girth from GENREG filename format (e.g., "12_3_3.scd" → n=12, k=3, girth=3)

- `parse_asc_file(filename)` - Moved from `spectral_gap_analysis.py` for centralized graph I/O

- `adjacency_to_edges(adj_dict)` - Moved from `spectral_gap_analysis.py`

- `shortcode_to_edges(code, n, k)` - Converts GENREG shortcode representation to edge list
  - Implements the algorithm described in GENREG documentation
  - Tracks vertex degrees to correctly reconstruct edges
  - Returns 0-indexed edge tuples as plain Python integers

- `parse_scd_file(filename, n=None, k=None)` - Main SCD parser with differential decompression
  - Auto-detects graph parameters from filename
  - Implements differential compression algorithm
  - Handles binary format with `numpy.uint8` arrays
  - Returns same format as ASC parser (list of edge lists)

- `load_graphs_from_file(filename)` - Format-agnostic graph loader (moved from `spectral_gap_analysis.py`)
  - Auto-detects file format (.asc or .scd) from extension
  - Recommended way to load graphs
  - Now available to all scripts in the project

### 2. Updated `spectral_gap_analysis.py`

**Changes:**
- Imported graph I/O functions from `aqc_spectral_utils` (including `load_graphs_from_file`)
- Removed duplicate `parse_asc_file()` and `adjacency_to_edges()` functions
- Removed duplicate `load_graphs_from_file()` function (now in utils)
- Updated `GENREG_FILES` to use `.scd` format for N=12: `{10: '10_3_3.asc', 12: '12_3_3.scd'}`
- Now uses centralized graph I/O from utilities module

### 3. Created `test_scd_parser.py`

**Validation Script:**
- Parses both `12_3_3.asc` and `12_3_3.scd`
- Validates graph count matches (85 graphs)
- Validates all edge lists match exactly (order-independent comparison)
- Validates graph properties (3-regular, 12 vertices, no duplicate edges)
- Displays sample graphs and comprehensive test results

## Key Technical Details

### SCD Binary Format

The `.scd` format uses **differential compression** to efficiently store multiple graphs:

1. **Shortcode Representation:**
   - For each vertex v (in order 1 to n), list only neighbors w where w > v
   - Example: Complete graph K4 → code `[2, 3, 4, 3, 4, 4]`
   - Avoids storing each edge twice

2. **Differential Compression:**
   - Each graph entry: `[common_prefix_length, new_values...]`
   - First graph: `0 2 3 4 3 4 4` (0 = no common prefix)
   - Subsequent graphs: reuse common prefix from previous graph
   - Example: If second graph shares first 6 values → `6 6 5 7 6 7 ...`

3. **Edge Reconstruction Algorithm:**
   ```python
   v = 0  # current vertex (0-indexed)
   for w_value in shortcode:
       w = w_value - 1  # convert to 0-indexed
       while degree[v] == k:
           v += 1  # advance to next unfilled vertex
       create_edge(v, w)
       degree[v] += 1
       degree[w] += 1
   ```

### Indexing Convention

- **GENREG files:** 1-indexed vertices
- **Our edge lists:** 0-indexed vertices (subtract 1 during conversion)
- Consistent with existing ASC parser implementation

## Validation Results

```
✅ ALL TESTS PASSED!
   • Parsed 85 graphs from both formats
   • All edge lists match exactly
   • All graphs validated as 3-regular on 12 vertices
```

### Test Statistics
- **File:** `12_3_3.scd` (12 vertices, 3-regular, girth 3)
- **Graphs parsed:** 85
- **Match rate:** 100% (all edge lists identical between ASC and SCD)
- **Property validation:** All graphs correctly have 18 edges (12 × 3 / 2)
- **Degree validation:** All 1020 vertices (85 × 12) have degree exactly 3

## Usage Examples

### Using SCD Files in Analysis

```python
from aqc_spectral_utils import parse_scd_file

# Auto-detect parameters from filename
graphs = parse_scd_file("graphs_rawdata/12_3_3.scd")
print(f"Loaded {len(graphs)} graphs")  # 85

# Or specify parameters explicitly
graphs = parse_scd_file("graphs_rawdata/12_3_3.scd", n=12, k=3)
```

### Format-Agnostic Loading

```python
from aqc_spectral_utils import load_graphs_from_file

# Automatically detects .asc or .scd format
graphs = load_graphs_from_file("graphs_rawdata/12_3_3.scd")  # Uses SCD parser
graphs = load_graphs_from_file("graphs_rawdata/10_3_3.asc")  # Uses ASC parser

# Also available through spectral_gap_analysis (re-exported)
from spectral_gap_analysis import load_graphs_from_file

graphs = load_graphs_from_file("graphs_rawdata/12_3_3.scd")
```

### Running the Validation Test

```bash
python3 test_scd_parser.py
```

## Benefits

1. **Binary Format:** `.scd` files are more compact than `.asc` files
2. **Efficiency:** Differential compression reduces file size for large graph databases
3. **Compatibility:** Seamlessly integrates with existing analysis pipeline
4. **Validation:** Comprehensive test suite ensures correctness
5. **Unified API:** Same function signatures and return types as ASC parser
6. **Centralized I/O:** All graph loading functions now in utilities module for reuse across scripts

## Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `aqc_spectral_utils.py` | Modified | Added ~180 lines of graph I/O code (including `load_graphs_from_file`) |
| `spectral_gap_analysis.py` | Modified | Removed ~80 lines duplicate code, now imports all graph I/O from utils |
| `test_scd_parser.py` | Created | Comprehensive validation test script |
| `SCD_IMPLEMENTATION_SUMMARY.md` | Created | This documentation |

## Future Enhancements

- Support for other GENREG file formats if needed
- Direct download and parsing from GENREG database URLs
- Performance benchmarking for large graph databases
- Caching mechanisms for frequently used graphs

