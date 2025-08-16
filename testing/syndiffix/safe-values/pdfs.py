import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

def create_salary_pdf_plot(data_df, sal_df, cell_sal_stitched_df):
    """
    Creates the salary distribution PDF plot comparing original and synthetic data.
    
    Args:
        data_df: Original data DataFrame
        sal_df: Synthetic salary data DataFrame  
        cell_sal_stitched_df: Stitched synthetic data DataFrame
        
    Returns:
        tuple: (sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched)
    """
    print("\n" + "="*60)
    print("=== CREATING SALARY PDF PLOT ===")
    
    # Extract and sort salary data
    original_sal = data_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    synthetic_sal = sal_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    stitched_sal = cell_sal_stitched_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    
    print(f"Original salary data:")
    print(f"  Min: {original_sal.min():,.0f}")
    print(f"  Max: {original_sal.max():,.0f}")
    print(f"  Mean: {original_sal.mean():,.0f}")
    print(f"  Zero values: {(original_sal == 0).sum()}")
    print(f"  Negative values: {(original_sal < 0).sum()}")
    print(f"  Positive values: {(original_sal > 0).sum()}")
    
    print(f"Synthetic salary data:")
    print(f"  Min: {synthetic_sal.min():,.0f}")
    print(f"  Max: {synthetic_sal.max():,.0f}")
    print(f"  Mean: {synthetic_sal.mean():,.0f}")
    print(f"  Zero values: {(synthetic_sal == 0).sum()}")
    print(f"  Negative values: {(synthetic_sal < 0).sum()}")
    print(f"  Positive values: {(synthetic_sal > 0).sum()}")
    
    print(f"Stitched salary data:")
    print(f"  Min: {stitched_sal.min():,.0f}")
    print(f"  Max: {stitched_sal.max():,.0f}")
    print(f"  Mean: {stitched_sal.mean():,.0f}")
    print(f"  Zero values: {(stitched_sal == 0).sum()}")
    print(f"  Negative values: {(stitched_sal < 0).sum()}")
    print(f"  Positive values: {(stitched_sal > 0).sum()}")
    
    # Compute Kolmogorov-Smirnov distance for salary data
    ks_statistic_sal, ks_pvalue_sal = stats.ks_2samp(original_sal, synthetic_sal)
    ks_statistic_sal_stitched, ks_pvalue_sal_stitched = stats.ks_2samp(original_sal, stitched_sal)
    
    print(f"\nKolmogorov-Smirnov test for salary data:")
    print(f"  Syn (salary only) KS distance: {ks_statistic_sal:.4f}")
    print(f"  Syn (salary only) p-value: {ks_pvalue_sal:.4e}")
    print(f"  Syn (all columns) KS distance: {ks_statistic_sal_stitched:.4f}")
    print(f"  Syn (all columns) p-value: {ks_pvalue_sal_stitched:.4e}")
    
    if ks_pvalue_sal < 0.001:
        print(f"  Result (salary only): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_sal < 0.05:
        print(f"  Result (salary only): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (salary only): No significant difference detected (p >= 0.05)")
        
    if ks_pvalue_sal_stitched < 0.001:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_sal_stitched < 0.05:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (all columns): No significant difference detected (p >= 0.05)")
    
    # Create salary distribution plot
    plt.figure(figsize=(7, 5))
    
    # Plot all three lines with dots at both ends
    plt.plot(range(len(original_sal)), original_sal, label='Original', color='blue', linewidth=3.0, alpha=0.8)
    plt.plot(0, original_sal.iloc[0], 'o', color='blue', markersize=6)  # Start dot
    plt.plot(len(original_sal)-1, original_sal.iloc[-1], 'o', color='blue', markersize=6)  # End dot
    
    plt.plot(range(len(synthetic_sal)), synthetic_sal, label='Syn (salary only)', color='red', linewidth=1.5, alpha=0.8)
    plt.plot(0, synthetic_sal.iloc[0], 'o', color='red', markersize=6)  # Start dot
    plt.plot(len(synthetic_sal)-1, synthetic_sal.iloc[-1], 'o', color='red', markersize=6)  # End dot
    
    plt.plot(range(len(stitched_sal)), stitched_sal, label='Syn (all columns)', color='green', linewidth=1.5, alpha=0.8)
    plt.plot(0, stitched_sal.iloc[0], 'o', color='green', markersize=6)  # Start dot
    plt.plot(len(stitched_sal)-1, stitched_sal.iloc[-1], 'o', color='green', markersize=6)  # End dot
    
    # Set symlog scale for y-axis (handles negative, zero, and positive values)
    plt.yscale('symlog', linthresh=1000)  # Linear threshold of 1000 for smooth transition
    
    # Remove the 10^2 and -10^2 tick labels to prevent overlap
    ax = plt.gca()
    yticks = ax.get_yticks()
    yticklabels = []
    for tick in yticks:
        if abs(tick) == 100:  # Remove 10^2 and -10^2 labels
            yticklabels.append('')
        else:
            yticklabels.append(ax.get_yticklabels()[list(yticks).index(tick)].get_text())
    ax.set_yticklabels(yticklabels)

    # Customize the plot
    plt.xlabel('Row Index (sorted by salary)', fontsize=12)
    plt.ylabel('Gesamtbetrag_Einkuenfte (symlog scale)', fontsize=12)
    plt.title('Salary Distribution: Original vs Synthetic\n(Sorted from smallest to largest)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    stats_text = []
    stats_text.append(f"Original: n={len(original_sal):,}, μ={original_sal.mean():,.0f}")
    stats_text.append(f"Syn (salary only): n={len(synthetic_sal):,}, μ={synthetic_sal.mean():,.0f}")
    stats_text.append(f"Syn (all columns): n={len(stitched_sal):,}, μ={stitched_sal.mean():,.0f}")
    stats_text.append(f"KS dist (salary only): {ks_statistic_sal:.4f}")
    stats_text.append(f"KS dist (all columns): {ks_statistic_sal_stitched:.4f}")
    
    plt.text(0.02, 0.98, '\n'.join(stats_text), transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10, family='monospace')
    
    # Save the plot
    sal_plot_path = os.path.join('salary_data', 'pdf_sal.png')
    plt.tight_layout()
    plt.savefig(sal_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salary PDF plot saved to: {sal_plot_path}")
    
    return sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched

def create_cell_count_pdf_plot(data_df, cell_cnt_df, cell_sal_stitched_df):
    """
    Creates the cell count distribution PDF plot comparing original and synthetic data.
    
    Args:
        data_df: Original data DataFrame
        cell_cnt_df: Synthetic cell count data DataFrame  
        cell_sal_stitched_df: Stitched synthetic data DataFrame
        
    Returns:
        tuple: (cell_cnt_plot_path, ks_statistic_cell, ks_pvalue_cell, ks_statistic_cell_stitched, ks_pvalue_cell_stitched)
    """
    print("\n" + "="*60)
    print("=== CREATING CELL COUNT PDF PLOT ===")
    
    # Extract and sort cell count data
    original_cell_cnt = data_df['cell_count'].sort_values().reset_index(drop=True)
    synthetic_cell_cnt = cell_cnt_df['cell_count'].sort_values().reset_index(drop=True)
    
    # Create cell_count for stitched data by counting Gitterzelle frequencies
    print("Creating cell_count for stitched data...")
    gitterzelle_counts_stitched = cell_sal_stitched_df['Gitterzelle'].value_counts()
    cell_sal_stitched_df_with_count = cell_sal_stitched_df.copy()
    cell_sal_stitched_df_with_count['cell_count'] = cell_sal_stitched_df_with_count['Gitterzelle'].map(gitterzelle_counts_stitched)
    stitched_cell_cnt = cell_sal_stitched_df_with_count['cell_count'].sort_values().reset_index(drop=True)
    
    print(f"Stitched cell count creation:")
    print(f"  Unique Gitterzelle values in stitched data: {len(gitterzelle_counts_stitched)}")
    print(f"  Cell count range: {stitched_cell_cnt.min()} to {stitched_cell_cnt.max()}")
    print(f"  Cell count mean: {stitched_cell_cnt.mean():.2f}")
    
    print(f"Original cell count data:")
    print(f"  Min: {original_cell_cnt.min()}")
    print(f"  Max: {original_cell_cnt.max()}")
    print(f"  Mean: {original_cell_cnt.mean():.2f}")
    print(f"  Unique values: {original_cell_cnt.nunique()}")
    
    print(f"Synthetic cell count data:")
    print(f"  Min: {synthetic_cell_cnt.min()}")
    print(f"  Max: {synthetic_cell_cnt.max()}")
    print(f"  Mean: {synthetic_cell_cnt.mean():.2f}")
    print(f"  Unique values: {synthetic_cell_cnt.nunique()}")
    
    print(f"Stitched cell count data:")
    print(f"  Min: {stitched_cell_cnt.min()}")
    print(f"  Max: {stitched_cell_cnt.max()}")
    print(f"  Mean: {stitched_cell_cnt.mean():.2f}")
    print(f"  Unique values: {stitched_cell_cnt.nunique()}")
    
    # Compute Kolmogorov-Smirnov distance for cell count data
    ks_statistic_cell, ks_pvalue_cell = stats.ks_2samp(original_cell_cnt, synthetic_cell_cnt)
    ks_statistic_cell_stitched, ks_pvalue_cell_stitched = stats.ks_2samp(original_cell_cnt, stitched_cell_cnt)
    
    print(f"\nKolmogorov-Smirnov test for cell count data:")
    print(f"  Syn (cell count only) KS distance: {ks_statistic_cell:.4f}")
    print(f"  Syn (cell count only) p-value: {ks_pvalue_cell:.4e}")
    print(f"  Syn (all columns) KS distance: {ks_statistic_cell_stitched:.4f}")
    print(f"  Syn (all columns) p-value: {ks_pvalue_cell_stitched:.4e}")
    
    if ks_pvalue_cell < 0.001:
        print(f"  Result (cell count only): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_cell < 0.05:
        print(f"  Result (cell count only): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (cell count only): No significant difference detected (p >= 0.05)")
        
    if ks_pvalue_cell_stitched < 0.001:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_cell_stitched < 0.05:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (all columns): No significant difference detected (p >= 0.05)")
    
    # Create cell count distribution plot
    plt.figure(figsize=(7, 5))
    
    # Plot all three lines with dots at both ends
    plt.plot(range(len(original_cell_cnt)), original_cell_cnt, label='Original', color='blue', linewidth=3.0, alpha=0.8)
    plt.plot(0, original_cell_cnt.iloc[0], 'o', color='blue', markersize=6)  # Start dot
    plt.plot(len(original_cell_cnt)-1, original_cell_cnt.iloc[-1], 'o', color='blue', markersize=6)  # End dot
    
    plt.plot(range(len(synthetic_cell_cnt)), synthetic_cell_cnt, label='Syn (cell count only)', color='orange', linewidth=1.5, alpha=0.8)
    plt.plot(0, synthetic_cell_cnt.iloc[0], 'o', color='orange', markersize=6)  # Start dot
    plt.plot(len(synthetic_cell_cnt)-1, synthetic_cell_cnt.iloc[-1], 'o', color='orange', markersize=6)  # End dot
    
    plt.plot(range(len(stitched_cell_cnt)), stitched_cell_cnt, label='Syn (all columns)', color='green', linewidth=1.5, alpha=0.8)
    plt.plot(0, stitched_cell_cnt.iloc[0], 'o', color='green', markersize=6)  # Start dot
    plt.plot(len(stitched_cell_cnt)-1, stitched_cell_cnt.iloc[-1], 'o', color='green', markersize=6)  # End dot
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Customize the plot
    plt.xlabel('Row Index (sorted by cell count)', fontsize=12)
    plt.ylabel('Cell Count (log scale)', fontsize=12)
    plt.title('Cell Count Distribution: Original vs Synthetic\n(Sorted from smallest to largest)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    stats_text = []
    stats_text.append(f"Original: n={len(original_cell_cnt):,}, μ={original_cell_cnt.mean():.1f}")
    stats_text.append(f"Syn (cell count only): n={len(synthetic_cell_cnt):,}, μ={synthetic_cell_cnt.mean():.1f}")
    stats_text.append(f"Syn (all columns): n={len(stitched_cell_cnt):,}, μ={stitched_cell_cnt.mean():.1f}")
    stats_text.append(f"Original range: [{original_cell_cnt.min()}, {original_cell_cnt.max()}]")
    stats_text.append(f"KS dist (cell count only): {ks_statistic_cell:.4f}")
    stats_text.append(f"KS dist (all columns): {ks_statistic_cell_stitched:.4f}")
    
    plt.text(0.02, 0.98, '\n'.join(stats_text), transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10, family='monospace')
    
    # Save the plot
    cell_cnt_plot_path = os.path.join('salary_data', 'pdf_cell_cnt.png')
    plt.tight_layout()
    plt.savefig(cell_cnt_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cell count PDF plot saved to: {cell_cnt_plot_path}")
    
    return cell_cnt_plot_path, ks_statistic_cell, ks_pvalue_cell, ks_statistic_cell_stitched, ks_pvalue_cell_stitched

def create_pdf_plots():
    """
    Creates two PDF plots comparing original and synthetic data distributions.
    
    1. pdf_sal.png: Salary distributions with symlog scale
    2. pdf_cell_cnt.png: Cell count distributions with log scale
    """
    
    # Define file paths
    data_csv_path = os.path.join('salary_data', 'data.csv')
    cell_cnt_parquet_path = os.path.join('salary_data', 'cell_cnt.parquet')
    sal_parquet_path = os.path.join('salary_data', 'sal.parquet')
    cell_sal_stitched_parquet_path = os.path.join('salary_data', 'cell_sal_stitched.parquet')
    
    # Check if all files exist
    files_to_check = [
        (data_csv_path, 'data.csv'),
        (cell_cnt_parquet_path, 'cell_cnt.parquet'),
        (sal_parquet_path, 'sal.parquet'),
        (cell_sal_stitched_parquet_path, 'cell_sal_stitched.parquet')
    ]
    
    for file_path, file_name in files_to_check:
        if not os.path.exists(file_path):
            print(f"Error: {file_name} not found at {file_path}")
            return
    
    print("Reading data files...")
    
    # Read the data files
    data_df = pd.read_csv(data_csv_path)
    cell_cnt_df = pd.read_parquet(cell_cnt_parquet_path)
    sal_df = pd.read_parquet(sal_parquet_path)
    cell_sal_stitched_df = pd.read_parquet(cell_sal_stitched_parquet_path)
    
    print(f"Data shapes:")
    print(f"  data.csv: {data_df.shape}")
    print(f"  cell_cnt.parquet: {cell_cnt_df.shape}")
    print(f"  sal.parquet: {sal_df.shape}")
    print(f"  cell_sal_stitched.parquet: {cell_sal_stitched_df.shape}")

    print("Unique Gitterzelle values:")
    print(f"  data.csv: {data_df['Gitterzelle'].nunique()}")
    print(f"  cell_sal_stitched.parquet: {cell_sal_stitched_df['Gitterzelle'].nunique()}")

    # Verify required columns exist
    if 'Gesamtbetrag_Einkuenfte' not in data_df.columns:
        print("Error: 'Gesamtbetrag_Einkuenfte' column not found in data.csv")
        return
    
    if 'Gesamtbetrag_Einkuenfte' not in sal_df.columns:
        print("Error: 'Gesamtbetrag_Einkuenfte' column not found in sal.parquet")
        return
    
    if 'cell_count' not in data_df.columns:
        print("Error: 'cell_count' column not found in data.csv")
        return
    
    if 'cell_count' not in cell_cnt_df.columns:
        print("Error: 'cell_count' column not found in cell_cnt.parquet")
        return
    
    if 'Gesamtbetrag_Einkuenfte' not in cell_sal_stitched_df.columns:
        print("Error: 'Gesamtbetrag_Einkuenfte' column not found in cell_sal_stitched.parquet")
        return
    
    if 'Gitterzelle' not in cell_sal_stitched_df.columns:
        print("Error: 'Gitterzelle' column not found in cell_sal_stitched.parquet")
        return
    
    # Create salary PDF plot
    sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched = create_salary_pdf_plot(
        data_df, sal_df, cell_sal_stitched_df
    )
    
    # Create cell count PDF plot
    cell_cnt_plot_path, ks_statistic_cell, ks_pvalue_cell, ks_statistic_cell_stitched, ks_pvalue_cell_stitched = create_cell_count_pdf_plot(
        data_df, cell_cnt_df, cell_sal_stitched_df
    )
    
    print("\n" + "="*60)
    print("=== PLOTS COMPLETE ===")
    print("Generated plots:")
    print(f"  1. {sal_plot_path}")
    print(f"  2. {cell_cnt_plot_path}")
    print(f"\nKolmogorov-Smirnov Summary:")
    print(f"  Salary data:")
    print(f"    Syn (salary only) KS distance: {ks_statistic_sal:.4f} (p={ks_pvalue_sal:.4e})")
    print(f"    Syn (all columns) KS distance: {ks_statistic_sal_stitched:.4f} (p={ks_pvalue_sal_stitched:.4e})")
    print(f"  Cell count data:")
    print(f"    Syn (cell count only) KS distance: {ks_statistic_cell:.4f} (p={ks_pvalue_cell:.4e})")
    print(f"    Syn (all columns) KS distance: {ks_statistic_cell_stitched:.4f} (p={ks_pvalue_cell_stitched:.4e})")
    print(f"\nNote: Lower KS distance indicates better match between distributions.")
    print(f"      KS distance ranges from 0 (perfect match) to 1 (completely different).")

if __name__ == "__main__":
    create_pdf_plots()
