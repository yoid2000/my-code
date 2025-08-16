import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

def create_income_pdf_plot(data_df, sal_df, cell_sal_stitched_df, dataset_name, output_dir):
    """
    Creates the income distribution PDF plot comparing original and synthetic data.
    
    Args:
        data_df: Original data DataFrame
        sal_df: Synthetic income data DataFrame  
        cell_sal_stitched_df: Stitched synthetic data DataFrame
        dataset_name: Name of dataset ('dense' or 'sparse')
        output_dir: Directory to save plots
        
    Returns:
        tuple: (sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched)
    """
    print(f"\n{'='*60}")
    print(f"=== CREATING {dataset_name.upper()} INCOME PDF PLOT ===")
    
    # Extract and sort income data
    original_sal = data_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    synthetic_sal = sal_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    stitched_sal = cell_sal_stitched_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    
    print(f"Original {dataset_name} income data:")
    print(f"  Min: {original_sal.min():,.0f}")
    print(f"  Max: {original_sal.max():,.0f}")
    print(f"  Mean: {original_sal.mean():,.0f}")
    print(f"  Zero values: {(original_sal == 0).sum()}")
    print(f"  Negative values: {(original_sal < 0).sum()}")
    print(f"  Positive values: {(original_sal > 0).sum()}")
    
    print(f"Synthetic {dataset_name} income data:")
    print(f"  Min: {synthetic_sal.min():,.0f}")
    print(f"  Max: {synthetic_sal.max():,.0f}")
    print(f"  Mean: {synthetic_sal.mean():,.0f}")
    print(f"  Zero values: {(synthetic_sal == 0).sum()}")
    print(f"  Negative values: {(synthetic_sal < 0).sum()}")
    print(f"  Positive values: {(synthetic_sal > 0).sum()}")
    
    print(f"Stitched {dataset_name} income data:")
    print(f"  Min: {stitched_sal.min():,.0f}")
    print(f"  Max: {stitched_sal.max():,.0f}")
    print(f"  Mean: {stitched_sal.mean():,.0f}")
    print(f"  Zero values: {(stitched_sal == 0).sum()}")
    print(f"  Negative values: {(stitched_sal < 0).sum()}")
    print(f"  Positive values: {(stitched_sal > 0).sum()}")
    
    # Compute Kolmogorov-Smirnov distance for income data
    ks_statistic_sal, ks_pvalue_sal = stats.ks_2samp(original_sal, synthetic_sal)
    ks_statistic_sal_stitched, ks_pvalue_sal_stitched = stats.ks_2samp(original_sal, stitched_sal)
    
    print(f"\nKolmogorov-Smirnov test for {dataset_name} income data:")
    print(f"  Syn (income only) KS distance: {ks_statistic_sal:.4f}")
    print(f"  Syn (income only) p-value: {ks_pvalue_sal:.4e}")
    print(f"  Syn (all columns) KS distance: {ks_statistic_sal_stitched:.4f}")
    print(f"  Syn (all columns) p-value: {ks_pvalue_sal_stitched:.4e}")
    
    if ks_pvalue_sal < 0.001:
        print(f"  Result (income only): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_sal < 0.05:
        print(f"  Result (income only): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (income only): No significant difference detected (p >= 0.05)")
        
    if ks_pvalue_sal_stitched < 0.001:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_sal_stitched < 0.05:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (all columns): No significant difference detected (p >= 0.05)")
    
    # Create income distribution plot
    plt.figure(figsize=(6, 3.5))
    
    # Plot all three lines with dots at both ends
    plt.plot(range(len(original_sal)), original_sal, label='Original', color='blue', linewidth=3.0, alpha=0.8)
    plt.plot(0, original_sal.iloc[0], 'o', color='blue', markersize=6)  # Start dot
    plt.plot(len(original_sal)-1, original_sal.iloc[-1], 'o', color='blue', markersize=6)  # End dot
    
    plt.plot(range(len(synthetic_sal)), synthetic_sal, label='Syn (income only)', color='orange', linewidth=1.5, alpha=0.8)
    plt.plot(0, synthetic_sal.iloc[0], 'o', color='orange', markersize=6)  # Start dot
    plt.plot(len(synthetic_sal)-1, synthetic_sal.iloc[-1], 'o', color='orange', markersize=6)  # End dot
    
    plt.plot(range(len(stitched_sal)), stitched_sal, label='Syn (income and cell)', color='green', linewidth=1.5, alpha=0.8)
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
    plt.xlabel('Sorted by income', fontsize=12)
    plt.ylabel('Income (symlog scale)', fontsize=12)
    # Remove x-axis tick labels
    plt.gca().set_xticklabels([])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    sal_plot_path = os.path.join(output_dir, f'pdf_sal_{dataset_name}.png')
    plt.savefig(sal_plot_path, dpi=300, bbox_inches='tight')
    sal_plot_path = os.path.join(output_dir, f'pdf_sal_{dataset_name}.pdf')
    plt.savefig(sal_plot_path)
    plt.close()
    print(f"{dataset_name.title()} income PDF plot saved to: {sal_plot_path}")
    
    return sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched

def create_grouped_income_pdf_plot(data_df, sal_df, cell_sal_stitched_df, dataset_name, output_dir):
    """
    Creates the grouped income distribution PDF plot comparing original and synthetic data.
    Each point represents the average of 10 consecutive sorted values.
    
    Args:
        data_df: Original data DataFrame
        sal_df: Synthetic income data DataFrame  
        cell_sal_stitched_df: Stitched synthetic data DataFrame
        dataset_name: Name of dataset ('dense' or 'sparse')
        output_dir: Directory to save plots
        
    Returns:
        tuple: (sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched)
    """
    print(f"\n{'='*60}")
    print(f"=== CREATING {dataset_name.upper()} GROUPED INCOME PDF PLOT ===")
    
    # Extract and sort income data
    original_sal = data_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    synthetic_sal = sal_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    stitched_sal = cell_sal_stitched_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
    
    def group_by_average(data, group_size=10):
        """Group data into averages of group_size consecutive values."""
        grouped_data = []
        for i in range(0, len(data), group_size):
            group = data[i:i+group_size]
            grouped_data.append(group.mean())
        return pd.Series(grouped_data)
    
    # Create grouped data (average of each 10 consecutive values)
    original_sal_grouped = group_by_average(original_sal)
    synthetic_sal_grouped = group_by_average(synthetic_sal)
    stitched_sal_grouped = group_by_average(stitched_sal)
    
    print(f"Original {dataset_name} income data (grouped):")
    print(f"  Original length: {len(original_sal):,} -> Grouped length: {len(original_sal_grouped):,}")
    print(f"  Min: {original_sal_grouped.min():,.0f}")
    print(f"  Max: {original_sal_grouped.max():,.0f}")
    print(f"  Mean: {original_sal_grouped.mean():,.0f}")
    
    print(f"Synthetic {dataset_name} income data (grouped):")
    print(f"  Original length: {len(synthetic_sal):,} -> Grouped length: {len(synthetic_sal_grouped):,}")
    print(f"  Min: {synthetic_sal_grouped.min():,.0f}")
    print(f"  Max: {synthetic_sal_grouped.max():,.0f}")
    print(f"  Mean: {synthetic_sal_grouped.mean():,.0f}")
    
    print(f"Stitched {dataset_name} income data (grouped):")
    print(f"  Original length: {len(stitched_sal):,} -> Grouped length: {len(stitched_sal_grouped):,}")
    print(f"  Min: {stitched_sal_grouped.min():,.0f}")
    print(f"  Max: {stitched_sal_grouped.max():,.0f}")
    print(f"  Mean: {stitched_sal_grouped.mean():,.0f}")
    
    # Compute Kolmogorov-Smirnov distance for grouped income data
    ks_statistic_sal, ks_pvalue_sal = stats.ks_2samp(original_sal_grouped, synthetic_sal_grouped)
    ks_statistic_sal_stitched, ks_pvalue_sal_stitched = stats.ks_2samp(original_sal_grouped, stitched_sal_grouped)
    
    print(f"\nKolmogorov-Smirnov test for {dataset_name} grouped income data:")
    print(f"  Syn (income only) KS distance: {ks_statistic_sal:.4f}")
    print(f"  Syn (income only) p-value: {ks_pvalue_sal:.4e}")
    print(f"  Syn (all columns) KS distance: {ks_statistic_sal_stitched:.4f}")
    print(f"  Syn (all columns) p-value: {ks_pvalue_sal_stitched:.4e}")
    
    if ks_pvalue_sal < 0.001:
        print(f"  Result (income only): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_sal < 0.05:
        print(f"  Result (income only): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (income only): No significant difference detected (p >= 0.05)")
        
    if ks_pvalue_sal_stitched < 0.001:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.001)")
    elif ks_pvalue_sal_stitched < 0.05:
        print(f"  Result (all columns): Distributions are significantly different (p < 0.05)")
    else:
        print(f"  Result (all columns): No significant difference detected (p >= 0.05)")
    
    # Create grouped income distribution plot
    plt.figure(figsize=(6, 3.5))
    
    # Plot all three lines with dots at both ends
    plt.plot(range(len(original_sal_grouped)), original_sal_grouped, label='Original', color='blue', linewidth=3.0, alpha=0.8)
    plt.plot(0, original_sal_grouped.iloc[0], 'o', color='blue', markersize=6)  # Start dot
    plt.plot(len(original_sal_grouped)-1, original_sal_grouped.iloc[-1], 'o', color='blue', markersize=6)  # End dot
    
    plt.plot(range(len(synthetic_sal_grouped)), synthetic_sal_grouped, label='Syn (income only)', color='orange', linewidth=1.5, alpha=0.8)
    plt.plot(0, synthetic_sal_grouped.iloc[0], 'o', color='orange', markersize=6)  # Start dot
    plt.plot(len(synthetic_sal_grouped)-1, synthetic_sal_grouped.iloc[-1], 'o', color='orange', markersize=6)  # End dot
    
    plt.plot(range(len(stitched_sal_grouped)), stitched_sal_grouped, label='Syn (all columns)', color='green', linewidth=1.5, alpha=0.8)
    plt.plot(0, stitched_sal_grouped.iloc[0], 'o', color='green', markersize=6)  # Start dot
    plt.plot(len(stitched_sal_grouped)-1, stitched_sal_grouped.iloc[-1], 'o', color='green', markersize=6)  # End dot
    
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
    plt.xlabel('Each point: avg of 10 sorted income values', fontsize=12)
    plt.ylabel('Income (symlog scale)', fontsize=12)
    # Remove x-axis tick labels
    plt.gca().set_xticklabels([])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    sal_plot_path = os.path.join(output_dir, f'pdf_sal_grouped_{dataset_name}.png')
    plt.savefig(sal_plot_path, dpi=300, bbox_inches='tight')
    sal_plot_path = os.path.join(output_dir, f'pdf_sal_grouped_{dataset_name}.pdf')
    plt.savefig(sal_plot_path)
    plt.close()
    print(f"{dataset_name.title()} grouped income PDF plot saved to: {sal_plot_path}")
    
    return sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched

def create_cell_count_pdf_plot(data_df, cell_cnt_df, cell_sal_stitched_df, dataset_name, output_dir):
    """
    Creates the cell count distribution PDF plot comparing original and synthetic data.
    
    Args:
        data_df: Original data DataFrame
        cell_cnt_df: Synthetic cell count data DataFrame  
        cell_sal_stitched_df: Stitched synthetic data DataFrame
        dataset_name: Name of dataset ('dense' or 'sparse')
        output_dir: Directory to save plots
        
    Returns:
        tuple: (cell_cnt_plot_path, ks_statistic_cell, ks_pvalue_cell, ks_statistic_cell_stitched, ks_pvalue_cell_stitched)
    """
    print(f"\n{'='*60}")
    print(f"=== CREATING {dataset_name.upper()} CELL COUNT PDF PLOT ===")
    
    # Extract and sort cell count data
    original_cell_cnt = data_df['cell_count'].sort_values().reset_index(drop=True)
    synthetic_cell_cnt = cell_cnt_df['cell_count'].sort_values().reset_index(drop=True)
    
    # Create cell_count for stitched data by counting Gitterzelle frequencies
    print(f"Creating cell_count for {dataset_name} stitched data...")
    gitterzelle_counts_stitched = cell_sal_stitched_df['Gitterzelle'].value_counts()
    cell_sal_stitched_df_with_count = cell_sal_stitched_df.copy()
    cell_sal_stitched_df_with_count['cell_count'] = cell_sal_stitched_df_with_count['Gitterzelle'].map(gitterzelle_counts_stitched)
    stitched_cell_cnt = cell_sal_stitched_df_with_count['cell_count'].sort_values().reset_index(drop=True)
    
    print(f"Stitched {dataset_name} cell count creation:")
    print(f"  Unique Gitterzelle values in stitched data: {len(gitterzelle_counts_stitched)}")
    print(f"  Cell count range: {stitched_cell_cnt.min()} to {stitched_cell_cnt.max()}")
    print(f"  Cell count mean: {stitched_cell_cnt.mean():.2f}")
    
    print(f"Original {dataset_name} cell count data:")
    print(f"  Min: {original_cell_cnt.min()}")
    print(f"  Max: {original_cell_cnt.max()}")
    print(f"  Mean: {original_cell_cnt.mean():.2f}")
    print(f"  Unique values: {original_cell_cnt.nunique()}")
    
    print(f"Synthetic {dataset_name} cell count data:")
    print(f"  Min: {synthetic_cell_cnt.min()}")
    print(f"  Max: {synthetic_cell_cnt.max()}")
    print(f"  Mean: {synthetic_cell_cnt.mean():.2f}")
    print(f"  Unique values: {synthetic_cell_cnt.nunique()}")
    
    print(f"Stitched {dataset_name} cell count data:")
    print(f"  Min: {stitched_cell_cnt.min()}")
    print(f"  Max: {stitched_cell_cnt.max()}")
    print(f"  Mean: {stitched_cell_cnt.mean():.2f}")
    print(f"  Unique values: {stitched_cell_cnt.nunique()}")
    
    # Compute Kolmogorov-Smirnov distance for cell count data
    ks_statistic_cell, ks_pvalue_cell = stats.ks_2samp(original_cell_cnt, synthetic_cell_cnt)
    ks_statistic_cell_stitched, ks_pvalue_cell_stitched = stats.ks_2samp(original_cell_cnt, stitched_cell_cnt)
    
    print(f"\nKolmogorov-Smirnov test for {dataset_name} cell count data:")
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
    plt.figure(figsize=(6, 3.5))
    
    # Plot all three lines with dots at both ends
    plt.plot(range(len(original_cell_cnt)), original_cell_cnt, label='Original', color='blue', linewidth=3.0, alpha=0.8)
    plt.plot(0, original_cell_cnt.iloc[0], 'o', color='blue', markersize=6)  # Start dot
    plt.plot(len(original_cell_cnt)-1, original_cell_cnt.iloc[-1], 'o', color='blue', markersize=6)  # End dot
    
    plt.plot(range(len(synthetic_cell_cnt)), synthetic_cell_cnt, label='Syn (pre-processed)', color='orange', linewidth=1.5, alpha=0.8)
    plt.plot(0, synthetic_cell_cnt.iloc[0], 'o', color='orange', markersize=6)  # Start dot
    plt.plot(len(synthetic_cell_cnt)-1, synthetic_cell_cnt.iloc[-1], 'o', color='orange', markersize=6)  # End dot
    
    plt.plot(range(len(stitched_cell_cnt)), stitched_cell_cnt, label='Syn (post-processed)', color='green', linewidth=1.5, alpha=0.8)
    plt.plot(0, stitched_cell_cnt.iloc[0], 'o', color='green', markersize=6)  # Start dot
    plt.plot(len(stitched_cell_cnt)-1, stitched_cell_cnt.iloc[-1], 'o', color='green', markersize=6)  # End dot
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Customize the plot
    plt.xlabel('Sorted by cell count', fontsize=12)
    plt.ylabel('Cell Count (log scale)', fontsize=12)
    # Remove x-axis tick labels
    plt.gca().set_xticklabels([])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    cell_cnt_plot_path = os.path.join(output_dir, f'pdf_cell_cnt_{dataset_name}.png')
    plt.savefig(cell_cnt_plot_path, dpi=300, bbox_inches='tight')
    cell_cnt_plot_path = os.path.join(output_dir, f'pdf_cell_cnt_{dataset_name}.pdf')
    plt.savefig(cell_cnt_plot_path)
    plt.close()
    print(f"{dataset_name.title()} cell count PDF plot saved to: {cell_cnt_plot_path}")
    
    return cell_cnt_plot_path, ks_statistic_cell, ks_pvalue_cell, ks_statistic_cell_stitched, ks_pvalue_cell_stitched

def process_dataset_plots(dataset_name, input_csv_path, output_dir):
    """
    Process plots for a single dataset (dense or sparse).
    
    Args:
        dataset_name: Name of dataset ('dense' or 'sparse')
        input_csv_path: Path to input CSV file
        output_dir: Directory containing parquet files and to save plots
        
    Returns:
        dict: Dictionary containing plot paths and KS statistics
    """
    print(f"\n{'='*80}")
    print(f"=== PROCESSING {dataset_name.upper()} DATASET PLOTS ===")
    print(f"Input CSV: {input_csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Define parquet file paths based on syn2.py naming convention
    cell_cnt_parquet_path = os.path.join(output_dir, f'cell_cnt_{dataset_name}.parquet')
    sal_parquet_path = os.path.join(output_dir, f'sal_{dataset_name}.parquet')
    cell_sal_stitched_parquet_path = os.path.join(output_dir, f'cell_sal_stitched_{dataset_name}.parquet')
    
    # Check if all files exist
    files_to_check = [
        (input_csv_path, f'data_{dataset_name}.csv'),
        (cell_cnt_parquet_path, f'cell_cnt_{dataset_name}.parquet'),
        (sal_parquet_path, f'sal_{dataset_name}.parquet'),
        (cell_sal_stitched_parquet_path, f'cell_sal_stitched_{dataset_name}.parquet')
    ]
    
    for file_path, file_name in files_to_check:
        if not os.path.exists(file_path):
            print(f"Error: {file_name} not found at {file_path}")
            return None
    
    print(f"Reading {dataset_name} data files...")
    
    # Read the data files
    data_df = pd.read_csv(input_csv_path)
    cell_cnt_df = pd.read_parquet(cell_cnt_parquet_path)
    sal_df = pd.read_parquet(sal_parquet_path)
    cell_sal_stitched_df = pd.read_parquet(cell_sal_stitched_parquet_path)
    
    print(f"{dataset_name.title()} data shapes:")
    print(f"  data_{dataset_name}.csv: {data_df.shape}")
    print(f"  cell_cnt_{dataset_name}.parquet: {cell_cnt_df.shape}")
    print(f"  sal_{dataset_name}.parquet: {sal_df.shape}")
    print(f"  cell_sal_stitched_{dataset_name}.parquet: {cell_sal_stitched_df.shape}")

    print(f"Unique Gitterzelle values in {dataset_name} data:")
    print(f"  data_{dataset_name}.csv: {data_df['Gitterzelle'].nunique()}")
    print(f"  cell_sal_stitched_{dataset_name}.parquet: {cell_sal_stitched_df['Gitterzelle'].nunique()}")

    # Verify required columns exist
    required_columns = [
        ('Gesamtbetrag_Einkuenfte', data_df, f'data_{dataset_name}.csv'),
        ('Gesamtbetrag_Einkuenfte', sal_df, f'sal_{dataset_name}.parquet'),
        ('cell_count', data_df, f'data_{dataset_name}.csv'),
        ('cell_count', cell_cnt_df, f'cell_cnt_{dataset_name}.parquet'),
        ('Gesamtbetrag_Einkuenfte', cell_sal_stitched_df, f'cell_sal_stitched_{dataset_name}.parquet'),
        ('Gitterzelle', cell_sal_stitched_df, f'cell_sal_stitched_{dataset_name}.parquet')
    ]
    
    for col_name, df, file_name in required_columns:
        if col_name not in df.columns:
            print(f"Error: '{col_name}' column not found in {file_name}")
            return None
    
    # Create income PDF plot
    sal_plot_path, ks_statistic_sal, ks_pvalue_sal, ks_statistic_sal_stitched, ks_pvalue_sal_stitched = create_income_pdf_plot(
        data_df, sal_df, cell_sal_stitched_df, dataset_name, output_dir
    )
    
    # Create grouped income PDF plot
    sal_grouped_plot_path, ks_statistic_sal_grouped, ks_pvalue_sal_grouped, ks_statistic_sal_stitched_grouped, ks_pvalue_sal_stitched_grouped = create_grouped_income_pdf_plot(
        data_df, sal_df, cell_sal_stitched_df, dataset_name, output_dir
    )
    
    # Create cell count PDF plot
    cell_cnt_plot_path, ks_statistic_cell, ks_pvalue_cell, ks_statistic_cell_stitched, ks_pvalue_cell_stitched = create_cell_count_pdf_plot(
        data_df, cell_cnt_df, cell_sal_stitched_df, dataset_name, output_dir
    )
    
    # Return results
    return {
        'dataset_name': dataset_name,
        'sal_plot_path': sal_plot_path,
        'sal_grouped_plot_path': sal_grouped_plot_path,
        'cell_cnt_plot_path': cell_cnt_plot_path,
        'ks_statistic_sal': ks_statistic_sal,
        'ks_pvalue_sal': ks_pvalue_sal,
        'ks_statistic_sal_stitched': ks_statistic_sal_stitched,
        'ks_pvalue_sal_stitched': ks_pvalue_sal_stitched,
        'ks_statistic_sal_grouped': ks_statistic_sal_grouped,
        'ks_pvalue_sal_grouped': ks_pvalue_sal_grouped,
        'ks_statistic_sal_stitched_grouped': ks_statistic_sal_stitched_grouped,
        'ks_pvalue_sal_stitched_grouped': ks_pvalue_sal_stitched_grouped,
        'ks_statistic_cell': ks_statistic_cell,
        'ks_pvalue_cell': ks_pvalue_cell,
        'ks_statistic_cell_stitched': ks_statistic_cell_stitched,
        'ks_pvalue_cell_stitched': ks_pvalue_cell_stitched
    }

def create_pdf_plots():
    """
    Creates six PDF plots comparing original and synthetic data distributions for both dense and sparse datasets.
    
    1. pdf_sal_dense.png: Dense income distributions with symlog scale
    2. pdf_sal_grouped_dense.png: Dense income distributions with grouped averages (10-value groups)
    3. pdf_cell_cnt_dense.png: Dense cell count distributions with log scale
    4. pdf_sal_sparse.png: Sparse income distributions with symlog scale
    5. pdf_sal_grouped_sparse.png: Sparse income distributions with grouped averages (10-value groups)
    6. pdf_cell_cnt_sparse.png: Sparse cell count distributions with log scale
    """
    
    print("="*80)
    print("=== DUAL DATASET PDF PLOT GENERATION ===")
    print("="*80)
    
    # Define input and output paths
    dense_input = os.path.join('salary_data2', 'data_dense.csv')
    sparse_input = os.path.join('salary_data2', 'data_sparse.csv')
    output_dir = 'salary_data2'
    
    print(f"Input files:")
    print(f"  Dense:  {dense_input}")
    print(f"  Sparse: {sparse_input}")
    print(f"Data directory: {output_dir}")
    
    # Process both datasets
    results = []
    
    # Process dense dataset
    dense_result = process_dataset_plots('dense', dense_input, output_dir)
    if dense_result:
        results.append(dense_result)
    
    # Process sparse dataset
    sparse_result = process_dataset_plots('sparse', sparse_input, output_dir)
    if sparse_result:
        results.append(sparse_result)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("=== PLOTS COMPLETE ===")
    print(f"{'='*80}")
    
    if results:
        print("Generated plots:")
        for i, result in enumerate(results, 1):
            dataset_name = result['dataset_name']
            print(f"  {i*3-2}. {result['sal_plot_path']}")
            print(f"  {i*3-1}. {result['sal_grouped_plot_path']}")
            print(f"  {i*3}. {result['cell_cnt_plot_path']}")
        
        print(f"\nKolmogorov-Smirnov Summary:")
        for result in results:
            dataset_name = result['dataset_name']
            print(f"  {dataset_name.title()} dataset:")
            print(f"    Income data (individual values):")
            print(f"      Syn (income only) KS distance: {result['ks_statistic_sal']:.4f} (p={result['ks_pvalue_sal']:.4e})")
            print(f"      Syn (all columns) KS distance: {result['ks_statistic_sal_stitched']:.4f} (p={result['ks_pvalue_sal_stitched']:.4e})")
            print(f"    Income data (grouped averages):")
            print(f"      Syn (income only) KS distance: {result['ks_statistic_sal_grouped']:.4f} (p={result['ks_pvalue_sal_grouped']:.4e})")
            print(f"      Syn (all columns) KS distance: {result['ks_statistic_sal_stitched_grouped']:.4f} (p={result['ks_pvalue_sal_stitched_grouped']:.4e})")
            print(f"    Cell count data:")
            print(f"      Syn (cell count only) KS distance: {result['ks_statistic_cell']:.4f} (p={result['ks_pvalue_cell']:.4e})")
            print(f"      Syn (all columns) KS distance: {result['ks_statistic_cell_stitched']:.4f} (p={result['ks_pvalue_cell_stitched']:.4e})")
        
        print(f"\nNote: Lower KS distance indicates better match between distributions.")
        print(f"      KS distance ranges from 0 (perfect match) to 1 (completely different).")
        
        print(f"\nTotal plots generated: {len(results) * 3}")
        
        # Generate summary tables
        print_summary_tables(results)
        
        # Generate LaTeX tables
        generate_latex_tables(results, output_dir)
    else:
        print("No plots were generated due to missing files or errors.")

def print_summary_tables(results):
    """
    Print summary tables for cell count and grouped income data,
    each containing both dense and sparse datasets.
    """
    # Organize results by dataset name for easier access
    results_dict = {result['dataset_name']: result for result in results}
    
    # Read all data files
    all_data = {}
    for dataset_name in ['dense', 'sparse']:
        if dataset_name in results_dict:
            # Define file paths
            input_csv_path = os.path.join('salary_data2', f'data_{dataset_name}.csv')
            cell_cnt_parquet_path = os.path.join('salary_data2', f'cell_cnt_{dataset_name}.parquet')
            sal_parquet_path = os.path.join('salary_data2', f'sal_{dataset_name}.parquet')
            cell_sal_stitched_parquet_path = os.path.join('salary_data2', f'cell_sal_stitched_{dataset_name}.parquet')
            
            # Read the data
            data_df = pd.read_csv(input_csv_path)
            cell_cnt_df = pd.read_parquet(cell_cnt_parquet_path)
            sal_df = pd.read_parquet(sal_parquet_path)
            cell_sal_stitched_df = pd.read_parquet(cell_sal_stitched_parquet_path)
            
            # Calculate cell count statistics
            original_cell_cnt = data_df['cell_count'].sort_values().reset_index(drop=True)
            synthetic_cell_cnt = cell_cnt_df['cell_count'].sort_values().reset_index(drop=True)
            
            # Create cell_count for stitched data
            gitterzelle_counts_stitched = cell_sal_stitched_df['Gitterzelle'].value_counts()
            cell_sal_stitched_df_with_count = cell_sal_stitched_df.copy()
            cell_sal_stitched_df_with_count['cell_count'] = cell_sal_stitched_df_with_count['Gitterzelle'].map(gitterzelle_counts_stitched)
            stitched_cell_cnt = cell_sal_stitched_df_with_count['cell_count'].sort_values().reset_index(drop=True)
            
            # Calculate grouped income statistics
            def group_by_average(data, group_size=10):
                grouped_data = []
                for i in range(0, len(data), group_size):
                    group = data[i:i+group_size]
                    grouped_data.append(group.mean())
                return pd.Series(grouped_data)
            
            original_sal = data_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
            synthetic_sal = sal_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
            stitched_sal = cell_sal_stitched_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
            
            original_sal_grouped = group_by_average(original_sal)
            synthetic_sal_grouped = group_by_average(synthetic_sal)
            stitched_sal_grouped = group_by_average(stitched_sal)
            
            all_data[dataset_name] = {
                'original_cell_cnt': original_cell_cnt,
                'synthetic_cell_cnt': synthetic_cell_cnt,
                'stitched_cell_cnt': stitched_cell_cnt,
                'original_sal_grouped': original_sal_grouped,
                'synthetic_sal_grouped': synthetic_sal_grouped,
                'stitched_sal_grouped': stitched_sal_grouped
            }
    
    # Print Cell Count Summary Table
    print(f"\n{'='*100}")
    print(f"=== CELL COUNT SUMMARY TABLE ===")
    print(f"{'='*100}")
    
    print(f"{'':<10} {'':<30} {'Min':<15} {'Max':<15} {'Median':<15} {'KS Score':<10}")
    print(f"{'-'*10} {'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
    
    for dataset_name in ['dense', 'sparse']:
        if dataset_name in all_data:
            data = all_data[dataset_name]
            result = results_dict[dataset_name]
            
            # Original data (show dataset name)
            print(f"{dataset_name.title():<10} {'Original':<30} {data['original_cell_cnt'].min():<15.0f} {data['original_cell_cnt'].max():<15.0f} {data['original_cell_cnt'].median():<15.0f} {'--':<10}")
            
            # Syn (cell count only) (no dataset name)
            ks_cell_only = result['ks_statistic_cell']
            print(f"{'':<10} {'Syn (cell count only)':<30} {data['synthetic_cell_cnt'].min():<15.0f} {data['synthetic_cell_cnt'].max():<15.0f} {data['synthetic_cell_cnt'].median():<15.0f} {ks_cell_only:<10.4f}")
            
            # Syn (all columns) (no dataset name)
            ks_cell_stitched = result['ks_statistic_cell_stitched']
            print(f"{'':<10} {'Syn (all columns)':<30} {data['stitched_cell_cnt'].min():<15.0f} {data['stitched_cell_cnt'].max():<15.0f} {data['stitched_cell_cnt'].median():<15.0f} {ks_cell_stitched:<10.4f}")
            
            if dataset_name == 'dense':
                print()  # Add spacing between dense and sparse
    
    # Print Grouped Income Summary Table
    print(f"\n{'='*100}")
    print(f"=== GROUPED INCOME SUMMARY TABLE ===")
    print(f"{'='*100}")
    
    print(f"{'':<10} {'':<30} {'Min':<15} {'Max':<15} {'Median':<15} {'KS Score':<10}")
    print(f"{'-'*10} {'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
    
    for dataset_name in ['dense', 'sparse']:
        if dataset_name in all_data:
            data = all_data[dataset_name]
            result = results_dict[dataset_name]
            
            # Original data (show dataset name)
            print(f"{dataset_name.title():<10} {'Original':<30} {data['original_sal_grouped'].min():<15.0f} {data['original_sal_grouped'].max():<15.0f} {data['original_sal_grouped'].median():<15.0f} {'--':<10}")
            
            # Syn (income only) (no dataset name)
            ks_sal_grouped_only = result['ks_statistic_sal_grouped']
            print(f"{'':<10} {'Syn (income only)':<30} {data['synthetic_sal_grouped'].min():<15.0f} {data['synthetic_sal_grouped'].max():<15.0f} {data['synthetic_sal_grouped'].median():<15.0f} {ks_sal_grouped_only:<10.4f}")
            
            # Syn (all columns) (no dataset name)
            ks_sal_grouped_stitched = result['ks_statistic_sal_stitched_grouped']
            print(f"{'':<10} {'Syn (all columns)':<30} {data['stitched_sal_grouped'].min():<15.0f} {data['stitched_sal_grouped'].max():<15.0f} {data['stitched_sal_grouped'].median():<15.0f} {ks_sal_grouped_stitched:<10.4f}")
            
            if dataset_name == 'dense':
                print()  # Add spacing between dense and sparse

def generate_latex_tables(results, output_dir):
    """
    Generate LaTeX tables for cell count and grouped income data,
    each containing both dense and sparse datasets.
    """
    # Organize results by dataset name for easier access
    results_dict = {result['dataset_name']: result for result in results}
    
    # Read all data files
    all_data = {}
    for dataset_name in ['dense', 'sparse']:
        if dataset_name in results_dict:
            # Define file paths
            input_csv_path = os.path.join(output_dir, f'data_{dataset_name}.csv')
            cell_cnt_parquet_path = os.path.join(output_dir, f'cell_cnt_{dataset_name}.parquet')
            sal_parquet_path = os.path.join(output_dir, f'sal_{dataset_name}.parquet')
            cell_sal_stitched_parquet_path = os.path.join(output_dir, f'cell_sal_stitched_{dataset_name}.parquet')
            
            # Read the data
            data_df = pd.read_csv(input_csv_path)
            cell_cnt_df = pd.read_parquet(cell_cnt_parquet_path)
            sal_df = pd.read_parquet(sal_parquet_path)
            cell_sal_stitched_df = pd.read_parquet(cell_sal_stitched_parquet_path)
            
            # Calculate cell count statistics
            original_cell_cnt = data_df['cell_count'].sort_values().reset_index(drop=True)
            synthetic_cell_cnt = cell_cnt_df['cell_count'].sort_values().reset_index(drop=True)
            
            # Create cell_count for stitched data
            gitterzelle_counts_stitched = cell_sal_stitched_df['Gitterzelle'].value_counts()
            cell_sal_stitched_df_with_count = cell_sal_stitched_df.copy()
            cell_sal_stitched_df_with_count['cell_count'] = cell_sal_stitched_df_with_count['Gitterzelle'].map(gitterzelle_counts_stitched)
            stitched_cell_cnt = cell_sal_stitched_df_with_count['cell_count'].sort_values().reset_index(drop=True)
            
            # Calculate grouped income statistics
            def group_by_average(data, group_size=10):
                grouped_data = []
                for i in range(0, len(data), group_size):
                    group = data[i:i+group_size]
                    grouped_data.append(group.mean())
                return pd.Series(grouped_data)
            
            original_sal = data_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
            synthetic_sal = sal_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
            stitched_sal = cell_sal_stitched_df['Gesamtbetrag_Einkuenfte'].sort_values().reset_index(drop=True)
            
            original_sal_grouped = group_by_average(original_sal)
            synthetic_sal_grouped = group_by_average(synthetic_sal)
            stitched_sal_grouped = group_by_average(stitched_sal)
            
            all_data[dataset_name] = {
                'original_cell_cnt': original_cell_cnt,
                'synthetic_cell_cnt': synthetic_cell_cnt,
                'stitched_cell_cnt': stitched_cell_cnt,
                'original_sal_grouped': original_sal_grouped,
                'synthetic_sal_grouped': synthetic_sal_grouped,
                'stitched_sal_grouped': stitched_sal_grouped
            }
    
    # Generate LaTeX file
    latex_path = os.path.join(output_dir, 'tables.tex')
    
    with open(latex_path, 'w') as f:
        # Write combined table
        f.write("% Combined Summary Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Data Distribution Summary}\n")
        f.write("\\label{tab:region}\n")
        f.write("\\begin{tabular}{llrrrl}\n")
        f.write("\\toprule\n")
        
        # Cell Counts section with heading in column header
        f.write("\\multicolumn{2}{c}{\\textbf{Cell Counts}} & Min & Max & Median & KS Score \\\\\n")
        f.write("\\midrule\n")
        
        for dataset_name in ['dense', 'sparse']:
            if dataset_name in all_data:
                data = all_data[dataset_name]
                result = results_dict[dataset_name]
                
                # Original data (show dataset name)
                f.write(f"{dataset_name.title()} & Original & {data['original_cell_cnt'].min():.0f} & {data['original_cell_cnt'].max():.0f} & {data['original_cell_cnt'].median():.0f} & -- \\\\\n")
                
                # Syn (cell count only) (no dataset name)
                ks_cell_only = result['ks_statistic_cell']
                f.write(f" & Syn (cell count only) & {data['synthetic_cell_cnt'].min():.0f} & {data['synthetic_cell_cnt'].max():.0f} & {data['synthetic_cell_cnt'].median():.0f} & {ks_cell_only:.4f} \\\\\n")
                
                # Syn (all columns) (no dataset name)
                ks_cell_stitched = result['ks_statistic_cell_stitched']
                f.write(f" & Syn (all columns) & {data['stitched_cell_cnt'].min():.0f} & {data['stitched_cell_cnt'].max():.0f} & {data['stitched_cell_cnt'].median():.0f} & {ks_cell_stitched:.4f} \\\\\n")
                
                if dataset_name == 'dense':
                    f.write("\\addlinespace\n")  # Add spacing between dense and sparse
        
        # Income section with heading in column header
        f.write("\\addlinespace\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{2}{c}{\\textbf{Income}} & Min & Max & Median & KS Score \\\\\n")
        f.write("\\midrule\n")
        
        for dataset_name in ['dense', 'sparse']:
            if dataset_name in all_data:
                data = all_data[dataset_name]
                result = results_dict[dataset_name]
                
                # Original data (show dataset name)
                f.write(f"{dataset_name.title()} & Original & {data['original_sal_grouped'].min():.0f} & {data['original_sal_grouped'].max():.0f} & {data['original_sal_grouped'].median():.0f} & -- \\\\\n")
                
                # Syn (income only) (no dataset name)
                ks_sal_grouped_only = result['ks_statistic_sal_grouped']
                f.write(f" & Syn (income only) & {data['synthetic_sal_grouped'].min():.0f} & {data['synthetic_sal_grouped'].max():.0f} & {data['synthetic_sal_grouped'].median():.0f} & {ks_sal_grouped_only:.4f} \\\\\n")
                
                # Syn (all columns) (no dataset name)
                ks_sal_grouped_stitched = result['ks_statistic_sal_stitched_grouped']
                f.write(f" & Syn (all columns) & {data['stitched_sal_grouped'].min():.0f} & {data['stitched_sal_grouped'].max():.0f} & {data['stitched_sal_grouped'].median():.0f} & {ks_sal_grouped_stitched:.4f} \\\\\n")
                
                if dataset_name == 'dense':
                    f.write("\\addlinespace\n")  # Add spacing between dense and sparse
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX tables saved to: {latex_path}")

if __name__ == "__main__":
    create_pdf_plots()
    create_pdf_plots()
