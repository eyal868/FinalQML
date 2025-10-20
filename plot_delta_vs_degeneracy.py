#!/usr/bin/env python3
"""
Script to plot the relationship between Delta_min and Max_degeneracy
from spectral gap analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_delta_vs_degeneracy(csv_file: str, output_dir: str = "outputs"):
    """
    Plot Delta_min vs Max_degeneracy from CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the data
    output_dir : str
        Directory to save the output plot
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Extract relevant columns
    delta_min = df['Delta_min']
    max_degeneracy = df['Max_degeneracy']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    scatter = ax.scatter(max_degeneracy, delta_min, 
                        alpha=0.7, 
                        s=100, 
                        c=delta_min,
                        cmap='viridis',
                        edgecolors='black',
                        linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Delta_min', fontsize=12)
    
    # Calculate and plot trend line
    z = np.polyfit(max_degeneracy, delta_min, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(max_degeneracy.min(), max_degeneracy.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
            label=f'Linear fit: y={z[0]:.4f}x+{z[1]:.4f}')
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(max_degeneracy, delta_min)[0, 1]
    
    # Labels and title
    ax.set_xlabel('Max Degeneracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('g_min', fontsize=14, fontweight='bold')
    ax.set_title('Spectral Gap vs MaxCut Degeneracy for N=12 3-Regular Graphs',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add correlation text
    ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}\nN = {len(df)} graphs', 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract filename info for output name
    csv_name = Path(csv_file).stem
    output_file = output_path / f"{csv_name}_delta_vs_degeneracy.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Display statistics
    print("\n=== Statistics ===")
    print(f"Number of graphs: {len(df)}")
    print(f"Delta_min range: [{delta_min.min():.4f}, {delta_min.max():.4f}]")
    print(f"Max_degeneracy range: [{max_degeneracy.min()}, {max_degeneracy.max()}]")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"Linear fit: Delta_min = {z[0]:.4f} * Max_degeneracy + {z[1]:.4f}")
    
    # Show plot
    plt.show()
    
    return fig, ax

def main():
    """Main execution function."""
    csv_file = "outputs/Delta_min_3_regular_N12_res20.csv"
    
    print(f"Loading data from: {csv_file}")
    plot_delta_vs_degeneracy(csv_file)

if __name__ == "__main__":
    main()

