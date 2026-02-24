#!/usr/bin/env python3
"""
Sample 100 rows from Delta_min_3_regular_N16_merged.csv with uniform distribution
across Delta_min value ranges.

Ensures graphs 744 and 3571 are included as edge cases.
"""

import pandas as pd
import numpy as np

np.random.seed(42)  # Reproducibility

# Load source data
df = pd.read_csv('outputs/Delta_min_3_regular_N16_merged.csv')

# Force include required graphs (edge cases)
required_ids = [744, 3571]
result = df[df['Graph_ID'].isin(required_ids)].copy()
remaining = df[~df['Graph_ID'].isin(required_ids)]

# Define bins from 0.20 to 0.80 in 0.05 increments (12 bins)
bins = [
    (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40),
    (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
    (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80)
]

# Target: 100 total rows, 2 already reserved for required graphs
target_total = 100 - len(required_ids)  # 98 remaining
target_per_bin = target_total // len(bins)  # 8 per bin

# First pass: sample up to target_per_bin from each bin
sampled_rows = []
leftover_quota = 0

for low, high in bins:
    bin_df = remaining[(remaining['Delta_min'] >= low) & (remaining['Delta_min'] < high)]
    n_available = len(bin_df)
    n_sample = min(n_available, target_per_bin)
    
    if n_available > 0:
        sampled_rows.append(bin_df.sample(n=n_sample))
    
    # Track any unused quota from small bins
    leftover_quota += target_per_bin - n_sample

# Combine first pass samples
if sampled_rows:
    first_pass = pd.concat(sampled_rows)
    result = pd.concat([result, first_pass])
    
    # Remove already sampled from remaining pool
    remaining = remaining[~remaining['Graph_ID'].isin(first_pass['Graph_ID'])]

# Second pass: distribute leftover quota to bins with remaining samples
if leftover_quota > 0:
    extra_samples = []
    for low, high in bins:
        if leftover_quota <= 0:
            break
        bin_df = remaining[(remaining['Delta_min'] >= low) & (remaining['Delta_min'] < high)]
        if len(bin_df) > 0:
            n_extra = min(len(bin_df), leftover_quota)
            extra_samples.append(bin_df.sample(n=n_extra))
            leftover_quota -= n_extra
            remaining = remaining[~remaining['Graph_ID'].isin(extra_samples[-1]['Graph_ID'])]
    
    if extra_samples:
        result = pd.concat([result] + extra_samples)

# Ensure exactly 100 rows
if len(result) > 100:
    # Keep required graphs, randomly drop extras
    required_rows = result[result['Graph_ID'].isin(required_ids)]
    other_rows = result[~result['Graph_ID'].isin(required_ids)]
    other_rows = other_rows.sample(n=100 - len(required_ids))
    result = pd.concat([required_rows, other_rows])
elif len(result) < 100:
    # Sample additional from remaining pool
    n_needed = 100 - len(result)
    extra = remaining.sample(n=min(n_needed, len(remaining)))
    result = pd.concat([result, extra])

# Sort by Delta_min for readability
result = result.sort_values('Delta_min').reset_index(drop=True)

# Save output
output_path = 'outputs/Delta_min_3_regular_N16_uniform100.csv'
result.to_csv(output_path, index=False)

# Print summary
print(f"Created {output_path} with {len(result)} rows")
print(f"\nRequired graphs included:")
print(f"  Graph 744: {744 in result['Graph_ID'].values} (Delta_min={result[result['Graph_ID']==744]['Delta_min'].values[0]:.4f})")
print(f"  Graph 3571: {3571 in result['Graph_ID'].values} (Delta_min={result[result['Graph_ID']==3571]['Delta_min'].values[0]:.4f})")

print(f"\nDelta_min range: {result['Delta_min'].min():.4f} to {result['Delta_min'].max():.4f}")

# Show distribution
print("\nDistribution by bin:")
for low, high in [(0.15, 0.20)] + bins + [(0.80, 0.85)]:
    count = len(result[(result['Delta_min'] >= low) & (result['Delta_min'] < high)])
    if count > 0:
        print(f"  {low:.2f}-{high:.2f}: {count} rows")




