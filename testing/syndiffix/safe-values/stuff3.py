import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from syndiffix import Synthesizer

def compute_relative_error(original_val, synthetic_val):
    """
    Compute relative error between synthetic and original values.
    Handles cases where original value is 0.
    """
    if original_val == 0:
        return 0 if synthetic_val == 0 else np.inf
    return (synthetic_val - original_val) / original_val

def add_fake_zip_column(df, dataset_name):
    """
    Add fake_zip column to dataframe if it doesn't exist.
    Number of distinct fake_zip values = len(df)/2000
    """
    if 'fake_zip' not in df.columns:
        print(f"  Adding fake_zip column to {dataset_name} dataset...")
        
        # Calculate number of distinct fake_zip values
        n_zip_values = max(1, len(df) // 2000)  # Ensure at least 1 zip value
        print(f"  Creating {n_zip_values} distinct fake_zip values for {len(df)} rows")
        
        # Generate fake_zip values as integers starting from 10000
        zip_values = list(range(10000, 10000 + n_zip_values))
        
        # Randomly assign fake_zip values to each row
        np.random.seed(42)  # For reproducibility
        df['fake_zip'] = np.random.choice(zip_values, size=len(df))
        
        print(f"  fake_zip column added with values ranging from {min(zip_values)} to {max(zip_values)}")
        return True
    else:
        print(f"  fake_zip column already exists in {dataset_name} dataset")
        return False

def create_synthetic_data(df, output_path, dataset_name):
    """
    Create synthetic data for Gesamtbetrag_Einkuenfte and fake_zip columns.
    """
    if os.path.exists(output_path):
        print(f"  Loading existing synthetic data from: {output_path}")
        syn_df = pd.read_parquet(output_path)
        print(f"  Loaded {len(syn_df)} rows")
    else:
        print(f"  Creating new synthetic data for {dataset_name}...")
        print(f"  Synthesizing columns: Gesamtbetrag_Einkuenfte, fake_zip")
        
        # Create synthesizer for the two columns
        synthesizer = Synthesizer(df[['Gesamtbetrag_Einkuenfte', 'fake_zip']])
        syn_df = synthesizer.sample()
        
        print(f"  Synthetic data created with {len(syn_df)} rows")
        
        # Save to parquet
        syn_df.to_parquet(output_path, index=False)
        print(f"  Saved synthetic data to: {output_path}")
    
    return syn_df

def compute_statistics_by_zip(df, dataset_name, data_type):
    """
    Compute statistics (sum, mean, median, std) of Gesamtbetrag_Einkuenfte by fake_zip.
    """
    print(f"  Computing statistics for {dataset_name} {data_type} data...")
    
    stats = df.groupby('fake_zip')['Gesamtbetrag_Einkuenfte'].agg([
        ('sum', 'sum'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std')
    ]).reset_index()
    
    # Handle NaN std values (when there's only one value in a group)
    stats['std'] = stats['std'].fillna(0)
    
    print(f"    {len(stats)} unique fake_zip values")
    print(f"    Statistics computed: sum, mean, median, std")
    
    return stats

def create_error_boxplot(orig_dense_stats, syn_dense_stats, orig_sparse_stats, syn_sparse_stats, output_path):
    """
    Create boxplot showing relative errors for all statistics.
    """
    print(f"\nCreating error boxplot...")
    
    # Merge original and synthetic statistics
    print("  Merging statistics...")
    
    # Dense dataset
    dense_merged = orig_dense_stats.merge(syn_dense_stats, on='fake_zip', suffixes=('_orig', '_syn'))
    
    # Sparse dataset  
    sparse_merged = orig_sparse_stats.merge(syn_sparse_stats, on='fake_zip', suffixes=('_orig', '_syn'))
    
    print(f"  Dense dataset: {len(dense_merged)} zip codes matched")
    print(f"  Sparse dataset: {len(sparse_merged)} zip codes matched")
    
    # Compute relative errors
    metrics = ['sum', 'mean', 'median', 'std']
    
    dense_errors = {}
    sparse_errors = {}
    
    for metric in metrics:
        orig_col = f'{metric}_orig'
        syn_col = f'{metric}_syn'
        
        # Dense errors
        dense_errors[metric] = [
            compute_relative_error(orig_val, syn_val) 
            for orig_val, syn_val in zip(dense_merged[orig_col], dense_merged[syn_col])
        ]
        
        # Sparse errors
        sparse_errors[metric] = [
            compute_relative_error(orig_val, syn_val) 
            for orig_val, syn_val in zip(sparse_merged[orig_col], sparse_merged[syn_col])
        ]
        
        # Remove infinite values for plotting
        dense_errors[metric] = [e for e in dense_errors[metric] if np.isfinite(e)]
        sparse_errors[metric] = [e for e in sparse_errors[metric] if np.isfinite(e)]
    
    # Create horizontal boxplot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for boxplot
    boxplot_data = []
    labels = []
    
    # Add data in order: dense sum, dense mean, dense median, dense std, sparse sum, sparse mean, sparse median, sparse std
    for dataset_name, errors_dict in [('Dense', dense_errors), ('Sparse', sparse_errors)]:
        for metric in metrics:
            boxplot_data.append(errors_dict[metric])
            labels.append(f'{dataset_name}\n{metric.title()}')
    
    # Create horizontal boxplot
    box_plot = ax.boxplot(boxplot_data, vert=False, patch_artist=True)
    
    # Color the boxes - different colors for dense vs sparse
    dense_color = '#1f77b4'  # Blue
    sparse_color = '#ff7f0e'  # Orange
    
    for i, patch in enumerate(box_plot['boxes']):
        if i < 4:  # Dense dataset (first 4 boxes)
            patch.set_facecolor(dense_color)
        else:  # Sparse dataset (last 4 boxes)
            patch.set_facecolor(sparse_color)
        patch.set_alpha(0.7)
    
    # Customize plot
    ax.set_yticklabels(labels)
    ax.set_xlabel('Relative Error', fontsize=12)
    ax.set_title('Relative Error Distribution by Dataset and Metric\n(Gesamtbetrag_Einkuenfte Statistics by fake_zip)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Set x-axis limits to -0.5 to 0.5
    ax.set_xlim(-0.5, 0.5)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Set y-axis limits to show all boxes clearly
    ax.set_ylim(0.5, len(labels) + 0.5)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Error boxplot saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n  Error Summary:")
    for dataset_name, errors_dict in [('Dense', dense_errors), ('Sparse', sparse_errors)]:
        print(f"    {dataset_name} dataset:")
        for metric in metrics:
            errors = errors_dict[metric]
            if len(errors) > 0:
                mean_error = np.mean(errors)
                median_error = np.median(errors)
                std_error = np.std(errors)
                print(f"      {metric.title()}: mean={mean_error:.4f}, median={median_error:.4f}, std={std_error:.4f} (n={len(errors)})")
            else:
                print(f"      {metric.title()}: no finite errors")

def main():
    """
    Main function to process datasets and create error analysis.
    """
    print("=" * 80)
    print("=== FAKE ZIP ERROR ANALYSIS ===")
    print("=" * 80)
    
    # Define paths
    data_dir = 'salary_data3'
    dense_csv_path = os.path.join(data_dir, 'data_dense.csv')
    sparse_csv_path = os.path.join(data_dir, 'data_sparse.csv')
    syn_dense_path = os.path.join(data_dir, 'syn_dense.parquet')
    syn_sparse_path = os.path.join(data_dir, 'syn_sparse.parquet')
    plot_path = os.path.join(data_dir, 'err.png')
    
    # Check if input files exist
    for file_path in [dense_csv_path, sparse_csv_path]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    print(f"Input files:")
    print(f"  Dense CSV: {dense_csv_path}")
    print(f"  Sparse CSV: {sparse_csv_path}")
    print(f"Output files:")
    print(f"  Synthetic dense: {syn_dense_path}")
    print(f"  Synthetic sparse: {syn_sparse_path}")
    print(f"  Error plot: {plot_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Read datasets
    print(f"\n" + "=" * 60)
    print("=== READING DATASETS ===")
    print("=" * 60)
    
    print(f"Reading dense dataset...")
    df_dense = pd.read_csv(dense_csv_path)
    print(f"  Dense dataset: {df_dense.shape[0]:,} rows × {df_dense.shape[1]} columns")
    
    print(f"Reading sparse dataset...")
    df_sparse = pd.read_csv(sparse_csv_path)
    print(f"  Sparse dataset: {df_sparse.shape[0]:,} rows × {df_sparse.shape[1]} columns")
    
    # Step 2: Add fake_zip columns if needed
    print(f"\n" + "=" * 60)
    print("=== ADDING FAKE_ZIP COLUMNS ===")
    print("=" * 60)
    
    dense_modified = add_fake_zip_column(df_dense, 'dense')
    sparse_modified = add_fake_zip_column(df_sparse, 'sparse')
    
    # Step 3: Write modified datasets back to CSV if they were modified
    if dense_modified or sparse_modified:
        print(f"\n" + "=" * 60)
        print("=== WRITING MODIFIED DATASETS ===")
        print("=" * 60)
        
        if dense_modified:
            print(f"Writing modified dense dataset to {dense_csv_path}...")
            df_dense.to_csv(dense_csv_path, index=False)
            print(f"  Dense dataset written with {len(df_dense)} rows")
        
        if sparse_modified:
            print(f"Writing modified sparse dataset to {sparse_csv_path}...")
            df_sparse.to_csv(sparse_csv_path, index=False)
            print(f"  Sparse dataset written with {len(df_sparse)} rows")
    
    # Step 4: Create synthetic datasets
    print(f"\n" + "=" * 60)
    print("=== CREATING SYNTHETIC DATASETS ===")
    print("=" * 60)
    
    print(f"Processing dense synthetic data...")
    syn_dense = create_synthetic_data(df_dense, syn_dense_path, 'dense')
    
    print(f"Processing sparse synthetic data...")
    syn_sparse = create_synthetic_data(df_sparse, syn_sparse_path, 'sparse')
    
    # Step 5: Compute statistics by fake_zip
    print(f"\n" + "=" * 60)
    print("=== COMPUTING STATISTICS BY FAKE_ZIP ===")
    print("=" * 60)
    
    # Original data statistics
    orig_dense_stats = compute_statistics_by_zip(df_dense, 'dense', 'original')
    orig_sparse_stats = compute_statistics_by_zip(df_sparse, 'sparse', 'original')
    
    # Synthetic data statistics
    syn_dense_stats = compute_statistics_by_zip(syn_dense, 'dense', 'synthetic')
    syn_sparse_stats = compute_statistics_by_zip(syn_sparse, 'sparse', 'synthetic')
    
    # Step 6: Create error boxplot
    print(f"\n" + "=" * 60)
    print("=== CREATING ERROR BOXPLOT ===")
    print("=" * 60)
    
    create_error_boxplot(orig_dense_stats, syn_dense_stats, orig_sparse_stats, syn_sparse_stats, plot_path)
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("=== PROCESSING COMPLETE ===")
    print("=" * 80)
    
    print(f"Dataset information:")
    print(f"  Dense dataset: {len(df_dense):,} rows, {df_dense['fake_zip'].nunique()} unique fake_zip values")
    print(f"  Sparse dataset: {len(df_sparse):,} rows, {df_sparse['fake_zip'].nunique()} unique fake_zip values")
    print(f"  Dense synthetic: {len(syn_dense):,} rows, {syn_dense['fake_zip'].nunique()} unique fake_zip values")
    print(f"  Sparse synthetic: {len(syn_sparse):,} rows, {syn_sparse['fake_zip'].nunique()} unique fake_zip values")
    
    print(f"\nOutput files created:")
    print(f"  {syn_dense_path}")
    print(f"  {syn_sparse_path}")
    print(f"  {plot_path}")

if __name__ == "__main__":
    main()
