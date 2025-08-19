import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_relative_error(original_val, synthetic_val):
    """
    Compute relative error between synthetic and original values.
    Handles cases where original value is 0.
    """
    if original_val == 0:
        return 0 if synthetic_val == 0 else np.inf
    return (synthetic_val - original_val) / original_val

def create_binned_boxplots(gitterzelle_stats, metric_name, output_dir):
    """
    Create binned boxplots for a given metric (freq, average, median, or sum).
    Creates 10 bins based on Gitterzelle frequency with roughly equal number of distinct cells per bin.
    """
    # Filter data for frequency > 10
    filtered_data = gitterzelle_stats[gitterzelle_stats['freq_orig'] > 10].copy()
    
    if len(filtered_data) == 0:
        print(f"No Gitterzelle values with frequency > 10 for {metric_name}")
        return
    
    print(f"Creating {metric_name} boxplot with {len(filtered_data)} Gitterzelle values (freq > 10)")
    
    # Sort by frequency and create 10 bins with roughly equal number of cells
    filtered_data = filtered_data.sort_values('freq_orig').reset_index(drop=True)
    
    # Create 10 bins with roughly equal number of cells
    n_bins = 10
    bin_size = len(filtered_data) // n_bins
    remainder = len(filtered_data) % n_bins
    
    # Assign bin labels
    bin_labels = []
    current_pos = 0
    
    for i in range(n_bins):
        # Add one extra cell to first 'remainder' bins to distribute remainder evenly
        current_bin_size = bin_size + (1 if i < remainder else 0)
        
        for j in range(current_bin_size):
            if current_pos < len(filtered_data):
                bin_labels.append(f"Bin {i+1}")
                current_pos += 1
    
    filtered_data['bin'] = bin_labels
    
    # Prepare data for boxplot
    boxplot_data = []
    bin_names = []
    bin_freq_ranges = []
    
    for i in range(n_bins):
        bin_name = f"Bin {i+1}"
        bin_data = filtered_data[filtered_data['bin'] == bin_name]
        
        if len(bin_data) > 0:
            # Get the relative errors for this bin
            errors = bin_data[f'{metric_name}_rel_error'].values
            # Remove infinite values for plotting
            finite_errors = errors[np.isfinite(errors)]
            
            if len(finite_errors) > 0:
                boxplot_data.append(finite_errors)
                bin_names.append(bin_name)
                
                # Calculate frequency range for this bin
                freq_min = bin_data['freq_orig'].min()
                freq_max = bin_data['freq_orig'].max()
                bin_freq_ranges.append(f"{freq_min}-{freq_max}")
    
    if len(boxplot_data) == 0:
        print(f"No valid data for {metric_name} boxplot")
        return
    
    # Create the boxplot
    plt.figure(figsize=(12, 6))
    
    box_plot = plt.boxplot(boxplot_data, labels=bin_names, patch_artist=True)
    
    # Customize boxplot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, patch in enumerate(box_plot['boxes']):
        color_idx = i % len(colors)
        patch.set_facecolor(colors[color_idx])
        patch.set_alpha(0.7)
    
    plt.title(f'Relative Error Distribution for {metric_name.title()} by Frequency Bins', fontsize=14)
    plt.xlabel('Frequency Bins (sorted by Gitterzelle frequency)', fontsize=12)
    plt.ylabel(f'Relative Error ({metric_name})', fontsize=12)
    
    # Add frequency range labels
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(1, len(bin_freq_ranges) + 1))
    ax2.set_xticklabels(bin_freq_ranges, rotation=45, ha='left')
    ax2.set_xlabel('Frequency Range', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'sal_error_{metric_name}_boxplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plot_path_pdf = os.path.join(output_dir, f'sal_error_{metric_name}_boxplot.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"{metric_name.title()} boxplot saved to: {plot_path}")
    
    # Print statistics for each bin
    print(f"\n{metric_name.title()} Error Statistics by Bin:")
    print(f"{'Bin':<8} {'Freq Range':<12} {'Count':<6} {'Mean Error':<12} {'Median Error':<14} {'Std Error':<12}")
    print("-" * 70)
    
    for i, (bin_name, freq_range) in enumerate(zip(bin_names, bin_freq_ranges)):
        if i < len(boxplot_data):
            errors = boxplot_data[i]
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            count = len(errors)
            
            print(f"{bin_name:<8} {freq_range:<12} {count:<6} {mean_error:<12.4f} {median_error:<14.4f} {std_error:<12.4f}")

def create_four_subplot_figure(gitterzelle_stats, output_dir):
    """
    Create a single figure with 4 subplots, one for each metric.
    """
    # Filter data for frequency > 10
    filtered_data = gitterzelle_stats[gitterzelle_stats['freq_orig'] > 10].copy()
    
    if len(filtered_data) == 0:
        print("No Gitterzelle values with frequency > 10")
        return
    
    print(f"Creating combined 4-subplot figure with {len(filtered_data)} Gitterzelle values (freq > 10)")
    
    # Sort by frequency and create 10 bins
    filtered_data = filtered_data.sort_values('freq_orig').reset_index(drop=True)
    
    # Create 10 bins with roughly equal number of cells
    n_bins = 10
    bin_size = len(filtered_data) // n_bins
    remainder = len(filtered_data) % n_bins
    
    # Assign bin labels
    bin_labels = []
    current_pos = 0
    
    for i in range(n_bins):
        current_bin_size = bin_size + (1 if i < remainder else 0)
        for j in range(current_bin_size):
            if current_pos < len(filtered_data):
                bin_labels.append(f"Bin {i+1}")
                current_pos += 1
    
    filtered_data['bin'] = bin_labels
    
    # Calculate bin frequency ranges for labeling
    bin_freq_ranges = []
    for i in range(n_bins):
        bin_name = f"Bin {i+1}"
        bin_data = filtered_data[filtered_data['bin'] == bin_name]
        if len(bin_data) > 0:
            freq_min = bin_data['freq_orig'].min()
            freq_max = bin_data['freq_orig'].max()
            bin_freq_ranges.append(f"{freq_min}-{freq_max}")
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['freq', 'average', 'median', 'sum']
    metric_titles = ['Frequency', 'Average', 'Median', 'Sum']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        
        # Prepare data for this metric
        boxplot_data = []
        bin_names = []
        
        for i in range(n_bins):
            bin_name = f"Bin {i+1}"
            bin_data = filtered_data[filtered_data['bin'] == bin_name]
            
            if len(bin_data) > 0:
                errors = bin_data[f'{metric}_rel_error'].values
                finite_errors = errors[np.isfinite(errors)]
                
                if len(finite_errors) > 0:
                    boxplot_data.append(finite_errors)
                    bin_names.append(str(i+1))  # Just use bin number
        
        if len(boxplot_data) > 0:
            # Create boxplot
            box_plot = ax.boxplot(boxplot_data, labels=bin_names, patch_artist=True)
            
            # Color the boxes
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for i, patch in enumerate(box_plot['boxes']):
                color_idx = i % len(colors)
                patch.set_facecolor(colors[color_idx])
                patch.set_alpha(0.7)
            
            ax.set_title(f'{title} Relative Error', fontsize=12)
            ax.set_xlabel('Frequency Bin', fontsize=10)
            ax.set_ylabel('Relative Error', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits to -2 to +2
            ax.set_ylim(-2, 2)
            
            # Rotate x-axis labels if needed
            if len(bin_names) > 5:
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'No valid data\nfor {title}', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{title} Relative Error', fontsize=12)
    
    plt.tight_layout()
    
    # Save the combined figure
    plot_path = os.path.join(output_dir, 'sal_error_combined_boxplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plot_path_pdf = os.path.join(output_dir, 'sal_error_combined_boxplots.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Combined 4-subplot figure saved to: {plot_path}")

def main():
    """
    Main function to analyze salary error between original and synthetic data.
    """
    print("=" * 80)
    print("=== SALARY ERROR ANALYSIS ===")
    print("=" * 80)
    
    # Define file paths
    data_dir = 'salary_data2'
    original_file = os.path.join(data_dir, 'data_dense.csv')
    synthetic_file = os.path.join(data_dir, 'cell_sal_stitched_dense.parquet')
    
    print(f"Reading original data from: {original_file}")
    print(f"Reading synthetic data from: {synthetic_file}")
    
    # Check if files exist
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        return
    
    if not os.path.exists(synthetic_file):
        print(f"Error: Synthetic file not found: {synthetic_file}")
        return
    
    # Read the data
    df_orig = pd.read_csv(original_file)
    df_syn = pd.read_parquet(synthetic_file)
    
    print(f"\nData shapes:")
    print(f"  Original: {df_orig.shape}")
    print(f"  Synthetic: {df_syn.shape}")
    
    print(f"\nColumn names:")
    print(f"  Original: {list(df_orig.columns)}")
    print(f"  Synthetic: {list(df_syn.columns)}")
    
    # Check required columns
    required_cols_orig = ['Gitterzelle', 'Gesamtbetrag_Einkuenfte']
    required_cols_syn = ['Gitterzelle', 'Gesamtbetrag_Einkuenfte']
    
    for col in required_cols_orig:
        if col not in df_orig.columns:
            print(f"Error: Column '{col}' not found in original data")
            return
    
    for col in required_cols_syn:
        if col not in df_syn.columns:
            print(f"Error: Column '{col}' not found in synthetic data")
            return
    
    print(f"\nUnique Gitterzelle values:")
    print(f"  Original: {df_orig['Gitterzelle'].nunique()}")
    print(f"  Synthetic: {df_syn['Gitterzelle'].nunique()}")
    
    # Group by Gitterzelle and compute statistics for original data
    print(f"\nComputing statistics for original data...")
    orig_stats = df_orig.groupby('Gitterzelle')['Gesamtbetrag_Einkuenfte'].agg([
        ('freq', 'count'),
        ('average', 'mean'),
        ('median', 'median'),
        ('sum', 'sum')
    ]).reset_index()
    
    # Group by Gitterzelle and compute statistics for synthetic data
    print(f"Computing statistics for synthetic data...")
    syn_stats = df_syn.groupby('Gitterzelle')['Gesamtbetrag_Einkuenfte'].agg([
        ('freq', 'count'),
        ('average', 'mean'),
        ('median', 'median'),
        ('sum', 'sum')
    ]).reset_index()
    
    print(f"\nStatistics computed:")
    print(f"  Original Gitterzelle groups: {len(orig_stats)}")
    print(f"  Synthetic Gitterzelle groups: {len(syn_stats)}")
    
    # Merge statistics and handle missing values
    print(f"\nMerging statistics...")
    
    # Start with original data
    merged_stats = orig_stats.copy()
    merged_stats.columns = ['Gitterzelle', 'freq_orig', 'average_orig', 'median_orig', 'sum_orig']
    
    # Add synthetic data
    syn_stats.columns = ['Gitterzelle', 'freq_syn', 'average_syn', 'median_syn', 'sum_syn']
    merged_stats = merged_stats.merge(syn_stats, on='Gitterzelle', how='left')
    
    # Fill missing synthetic values with 0
    syn_cols = ['freq_syn', 'average_syn', 'median_syn', 'sum_syn']
    for col in syn_cols:
        merged_stats[col] = merged_stats[col].fillna(0)
    
    print(f"Merged statistics shape: {merged_stats.shape}")
    
    # Count missing synthetic values
    missing_count = 0
    for col in syn_cols:
        missing_count += merged_stats[col].isnull().sum()
    print(f"Missing synthetic values: {missing_count}")
    
    # Compute relative errors
    print(f"\nComputing relative errors...")
    
    metrics = ['freq', 'average', 'median', 'sum']
    for metric in metrics:
        orig_col = f'{metric}_orig'
        syn_col = f'{metric}_syn'
        error_col = f'{metric}_rel_error'
        
        merged_stats[error_col] = merged_stats.apply(
            lambda row: compute_relative_error(row[orig_col], row[syn_col]), axis=1
        )
    
    print(f"Relative errors computed for metrics: {metrics}")
    
    # Print summary statistics
    print(f"\n" + "=" * 60)
    print("=== SUMMARY STATISTICS ===")
    print("=" * 60)
    
    for metric in metrics:
        error_col = f'{metric}_rel_error'
        errors = merged_stats[error_col]
        finite_errors = errors[np.isfinite(errors)]
        
        print(f"\n{metric.upper()} relative errors:")
        print(f"  Total Gitterzelle values: {len(errors)}")
        print(f"  Finite errors: {len(finite_errors)}")
        print(f"  Infinite errors: {len(errors) - len(finite_errors)}")
        
        if len(finite_errors) > 0:
            print(f"  Mean: {finite_errors.mean():.4f}")
            print(f"  Median: {finite_errors.median():.4f}")
            print(f"  Std: {finite_errors.std():.4f}")
            print(f"  Min: {finite_errors.min():.4f}")
            print(f"  Max: {finite_errors.max():.4f}")
    
    # Filter for frequency > 10 and show distribution
    freq_filtered = merged_stats[merged_stats['freq_orig'] > 10]
    print(f"\nGitterzelle values with frequency > 10: {len(freq_filtered)} out of {len(merged_stats)}")
    
    if len(freq_filtered) > 0:
        print(f"Frequency range for filtered data: {freq_filtered['freq_orig'].min()} to {freq_filtered['freq_orig'].max()}")
        
        # Create individual boxplots for each metric
        print(f"\n" + "=" * 60)
        print("=== CREATING INDIVIDUAL BOXPLOTS ===")
        print("=" * 60)
        
        for metric in metrics:
            create_binned_boxplots(merged_stats, metric, data_dir)
        
        # Create combined 4-subplot figure
        print(f"\n" + "=" * 60)
        print("=== CREATING COMBINED 4-SUBPLOT FIGURE ===")
        print("=" * 60)
        
        create_four_subplot_figure(merged_stats, data_dir)
        
        # Save the merged statistics for further analysis
        output_stats_file = os.path.join(data_dir, 'sal_error_statistics.csv')
        merged_stats.to_csv(output_stats_file, index=False)
        print(f"\nMerged statistics saved to: {output_stats_file}")
        
    else:
        print("No Gitterzelle values with frequency > 10 found. Cannot create boxplots.")
    
    print(f"\n" + "=" * 80)
    print("=== SALARY ERROR ANALYSIS COMPLETE ===")
    print("=" * 80)

if __name__ == "__main__":
    main()
