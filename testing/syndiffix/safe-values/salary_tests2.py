import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde
from syndiffix import Synthesizer
from syndiffix.stitcher import stitch

# Read the salary data CSV files
df1 = pd.read_csv('salary_data2/data1.csv')
df2 = pd.read_csv('salary_data2/data2.csv')

# Display column names and number of rows for both datasets
print("=== ORIGINAL DATA ANALYSIS ===")
print("Dataset 1 (data1.csv):")
print("Column names:")
for i, col in enumerate(df1.columns):
    print(f"  {i+1}. {col}")

print(f"\nNumber of rows: {len(df1)}")
print(f"Number of columns: {len(df1.columns)}")

print("\nDataset 2 (data2.csv):")
print("Column names:")
for i, col in enumerate(df2.columns):
    print(f"  {i+1}. {col}")

print(f"\nNumber of rows: {len(df2)}")
print(f"Number of columns: {len(df2.columns)}")

# Display basic info about both dataframes
print("\nDataset 1 info:")
print(df1.info())

print("\nDataset 2 info:")
print(df2.info())

# Show first few rows for both datasets
print("\nDataset 1 - First 5 rows:")
print(df1.head())

print("\nDataset 2 - First 5 rows:")
print(df2.head())

print("\n" + "="*60)
print("=== SYNTHESIZING DATAFRAMES ===")

# Function to create parquet files for both datasets
def create_synthetic_data(df, dataset_name, prefix):
    # Define parquet file paths
    df_cell_path = os.path.join('salary_data2', f'{prefix}_cell.parquet')
    df_cell_sal_path = os.path.join('salary_data2', f'{prefix}_cell_sal.parquet')
    df_sal_path = os.path.join('salary_data2', f'{prefix}_sal.parquet')
    df_cell_sal_stitched_path = os.path.join('salary_data2', f'{prefix}_cell_sal_stitched.parquet')
    
    print(f"\n=== Processing {dataset_name} ===")
    
    # 1. df_cell - synthesizes column Gitterzelle only with safe-values option
    print(f"\n1. Processing {prefix}_cell (Gitterzelle only, safe-values)...")
    if os.path.exists(df_cell_path):
        print(f"   Loading existing {prefix}_cell from: {df_cell_path}")
        df_cell = pd.read_parquet(df_cell_path)
        print(f"   {prefix}_cell loaded with {len(df_cell)} rows")
    else:
        print(f"   Creating new {prefix}_cell...")
        df_cell = Synthesizer(df[['Gitterzelle']], value_safe_columns=['Gitterzelle']).sample()
        print(f"   {prefix}_cell created with {len(df_cell)} rows")
        df_cell.to_parquet(df_cell_path)
        print(f"   {prefix}_cell saved to: {df_cell_path}")

    # 2. df_cell_sal - synthesizes Gitterzelle and Gesamtbetrag_Einkuenfte with safe-values for Gitterzelle
    print(f"\n2. Processing {prefix}_cell_sal (Gitterzelle + Gesamtbetrag_Einkuenfte, safe-values for Gitterzelle)...")
    if os.path.exists(df_cell_sal_path):
        print(f"   Loading existing {prefix}_cell_sal from: {df_cell_sal_path}")
        df_cell_sal = pd.read_parquet(df_cell_sal_path)
        print(f"   {prefix}_cell_sal loaded with {len(df_cell_sal)} rows")
    else:
        print(f"   Creating new {prefix}_cell_sal...")
        df_cell_sal = Synthesizer(df[['Gitterzelle', 'Gesamtbetrag_Einkuenfte']], value_safe_columns=['Gitterzelle']).sample()
        print(f"   {prefix}_cell_sal created with {len(df_cell_sal)} rows")
        df_cell_sal.to_parquet(df_cell_sal_path)
        print(f"   {prefix}_cell_sal saved to: {df_cell_sal_path}")

    # 3. df_sal - synthesizes Gesamtbetrag_Einkuenfte only with no safe-values
    print(f"\n3. Processing {prefix}_sal (Gesamtbetrag_Einkuenfte only, no safe-values)...")
    if os.path.exists(df_sal_path):
        print(f"   Loading existing {prefix}_sal from: {df_sal_path}")
        df_sal = pd.read_parquet(df_sal_path)
        print(f"   {prefix}_sal loaded with {len(df_sal)} rows")
    else:
        print(f"   Creating new {prefix}_sal...")
        df_sal = Synthesizer(df[['Gesamtbetrag_Einkuenfte']]).sample()
        print(f"   {prefix}_sal created with {len(df_sal)} rows")
        df_sal.to_parquet(df_sal_path)
        print(f"   {prefix}_sal saved to: {df_sal_path}")

    # 4. df_cell_sal_stitched - stitched version of df_cell_sal
    print(f"\n4. Processing {prefix}_cell_sal_stitched (stitched version)...")
    if os.path.exists(df_cell_sal_stitched_path):
        print(f"   Loading existing {prefix}_cell_sal_stitched from: {df_cell_sal_stitched_path}")
        df_cell_sal_stitched = pd.read_parquet(df_cell_sal_stitched_path)
        print(f"   {prefix}_cell_sal_stitched loaded with {len(df_cell_sal_stitched)} rows")
    else:
        print(f"   Creating new {prefix}_cell_sal_stitched...")
        df_cell_sal_stitched = stitch(df_left=df_cell, df_right=df_cell_sal, shared=False)
        print(f"   {prefix}_cell_sal_stitched created with {len(df_cell_sal_stitched)} rows")
        df_cell_sal_stitched.to_parquet(df_cell_sal_stitched_path)
        print(f"   {prefix}_cell_sal_stitched saved to: {df_cell_sal_stitched_path}")
    
    return df_cell, df_cell_sal, df_sal, df_cell_sal_stitched

# Create synthetic data for both datasets
d1_cell, d1_cell_sal, d1_sal, d1_cell_sal_stitched = create_synthetic_data(df1, "Dataset 1", "d1")
d2_cell, d2_cell_sal, d2_sal, d2_cell_sal_stitched = create_synthetic_data(df2, "Dataset 2", "d2")

print("\n" + "="*60)
print("=== DATAFRAMES SUMMARY ===")

# Function to display dataframe stats
def display_dataframe_stats(name, dataframe):
    print(f"\n{name}:")
    print(f"  Shape: {dataframe.shape}")
    print(f"  Columns: {list(dataframe.columns)}")
    print(f"  First few rows:")
    print(dataframe.head())

# Display stats for both datasets and their synthetic versions
display_dataframe_stats("Original Dataset 1 (d1)", df1)
display_dataframe_stats("D1 Cell (safe Gitterzelle)", d1_cell)
display_dataframe_stats("D1 Cell+Sal (safe Gitterzelle)", d1_cell_sal)
display_dataframe_stats("D1 Sal (no safe-values)", d1_sal)
display_dataframe_stats("D1 Cell+Sal Stitched", d1_cell_sal_stitched)

display_dataframe_stats("Original Dataset 2 (d2)", df2)
display_dataframe_stats("D2 Cell (safe Gitterzelle)", d2_cell)
display_dataframe_stats("D2 Cell+Sal (safe Gitterzelle)", d2_cell_sal)
display_dataframe_stats("D2 Sal (no safe-values)", d2_sal)
display_dataframe_stats("D2 Cell+Sal Stitched", d2_cell_sal_stitched)

print("\n" + "="*60)
print("=== COMPARISON SUMMARY ===")

print("\nDataframe sizes comparison:")
print(f"Original Dataset 1:      {df1.shape[0]:,} rows × {df1.shape[1]} columns")
print(f"D1 Cell:                 {d1_cell.shape[0]:,} rows × {d1_cell.shape[1]} columns")
print(f"D1 Cell+Sal:             {d1_cell_sal.shape[0]:,} rows × {d1_cell_sal.shape[1]} columns")
print(f"D1 Sal:                  {d1_sal.shape[0]:,} rows × {d1_sal.shape[1]} columns")
print(f"D1 Cell+Sal Stitched:    {d1_cell_sal_stitched.shape[0]:,} rows × {d1_cell_sal_stitched.shape[1]} columns")

print(f"\nOriginal Dataset 2:      {df2.shape[0]:,} rows × {df2.shape[1]} columns")
print(f"D2 Cell:                 {d2_cell.shape[0]:,} rows × {d2_cell.shape[1]} columns")
print(f"D2 Cell+Sal:             {d2_cell_sal.shape[0]:,} rows × {d2_cell_sal.shape[1]} columns")
print(f"D2 Sal:                  {d2_sal.shape[0]:,} rows × {d2_sal.shape[1]} columns")
print(f"D2 Cell+Sal Stitched:    {d2_cell_sal_stitched.shape[0]:,} rows × {d2_cell_sal_stitched.shape[1]} columns")

print("\nUnique Gitterzelle values:")
print(f"Original Dataset 1:      {df1['Gitterzelle'].nunique():,}")
print(f"D1 Cell:                 {d1_cell['Gitterzelle'].nunique():,}")
print(f"D1 Cell+Sal:             {d1_cell_sal['Gitterzelle'].nunique():,}")
print(f"D1 Cell+Sal Stitched:    {d1_cell_sal_stitched['Gitterzelle'].nunique():,}")

print(f"\nOriginal Dataset 2:      {df2['Gitterzelle'].nunique():,}")
print(f"D2 Cell:                 {d2_cell['Gitterzelle'].nunique():,}")
print(f"D2 Cell+Sal:             {d2_cell_sal['Gitterzelle'].nunique():,}")
print(f"D2 Cell+Sal Stitched:    {d2_cell_sal_stitched['Gitterzelle'].nunique():,}")

print("\n" + "="*60)
print("=== CREATING PLOTS ===")

# Count occurrences of each Gitterzelle value for both datasets
d1_original_counts = df1['Gitterzelle'].value_counts()
d1_cell_counts = d1_cell['Gitterzelle'].value_counts()

d2_original_counts = df2['Gitterzelle'].value_counts()  
d2_cell_counts = d2_cell['Gitterzelle'].value_counts()

# Determine the range for smooth curves and create shared functions
all_counts = [d1_original_counts, d1_cell_counts, d2_original_counts, d2_cell_counts]
min_count = min([counts.min() for counts in all_counts])
max_count = max([counts.max() for counts in all_counts])

# Create x-axis values for smooth curves (linear spacing)
x_smooth = np.linspace(min_count, max_count, 1000)

# Function to calculate KDE on linear scale
def linear_kde(data, x_eval):
    kde = gaussian_kde(data)
    return kde(x_eval)

print("\n" + "="*50)
print("=== CREATING CLEAN GITTERZELLE COUNT DISTRIBUTION PLOT ===")

# Updated count_dist_pdf_clean.png with both datasets (4 lines total)
count_clean_plot_path = os.path.join('salary_data2', 'count_dist_pdf_clean.png')
if os.path.exists(count_clean_plot_path):
    print(f"Clean Gitterzelle count distribution plot already exists at: {count_clean_plot_path} (skipping creation)")
else:
    print("Creating clean Gitterzelle count distribution plot with both datasets...")
    plt.figure(figsize=(10, 6))

    print(f"Count statistics:")
    print(f"D1 Original: min={d1_original_counts.min()}, max={d1_original_counts.max()}, mean={d1_original_counts.mean():.2f}")
    print(f"D1 Cell: min={d1_cell_counts.min()}, max={d1_cell_counts.max()}, mean={d1_cell_counts.mean():.2f}")
    print(f"D2 Original: min={d2_original_counts.min()}, max={d2_original_counts.max()}, mean={d2_original_counts.mean():.2f}")
    print(f"D2 Cell: min={d2_cell_counts.min()}, max={d2_cell_counts.max()}, mean={d2_cell_counts.mean():.2f}")

    # Plot curves - 4 lines total (2 for each dataset)
    # D1 Original (dashed, thicker, plotted first)
    d1_original_density = linear_kde(d1_original_counts.values, x_smooth)
    plt.plot(x_smooth, d1_original_density, label='D1 Original', color='blue', 
             linewidth=2.5, linestyle='--')

    # D1 Syn, location only
    d1_cell_density = linear_kde(d1_cell_counts.values, x_smooth)
    plt.plot(x_smooth, d1_cell_density, label='D1 Syn, location only', 
             color='red', linewidth=2)

    # D2 Original (dashed, thicker)
    d2_original_density = linear_kde(d2_original_counts.values, x_smooth)
    plt.plot(x_smooth, d2_original_density, label='D2 Original', color='green', 
             linewidth=2.5, linestyle='--')

    # D2 Syn, location only
    d2_cell_density = linear_kde(d2_cell_counts.values, x_smooth)
    plt.plot(x_smooth, d2_cell_density, label='D2 Syn, location only', 
             color='orange', linewidth=2)

    # Customize the plot
    plt.xlabel('Cell Count', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    # No title as requested
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set specific x-axis ticks
    plt.xticks([1, 2, 5, 10, 20, 50, 100, 200])

    # Add statistics text box in lower left
    text_info = []
    text_info.append(f"D1 Original: μ={d1_original_counts.mean():.1f}, σ={d1_original_counts.std():.1f}, n={len(d1_original_counts)}")
    text_info.append(f"D1 Syn, loc only: μ={d1_cell_counts.mean():.1f}, σ={d1_cell_counts.std():.1f}, n={len(d1_cell_counts)}")
    text_info.append(f"D2 Original: μ={d2_original_counts.mean():.1f}, σ={d2_original_counts.std():.1f}, n={len(d2_original_counts)}")
    text_info.append(f"D2 Syn, loc only: μ={d2_cell_counts.mean():.1f}, σ={d2_cell_counts.std():.1f}, n={len(d2_cell_counts)}")

    # Position text box in lower left
    plt.text(0.02, 0.02, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=7, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(count_clean_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clean Gitterzelle count distribution plot saved to: {count_clean_plot_path}")

# Function to create sal_dist_pdf_clean.png for a dataset
def create_sal_dist_clean_plot(df_orig, df_sal, df_cell_sal_stitched, dataset_name, prefix):
    print(f"\n=== CREATING {dataset_name.upper()} SALARY DISTRIBUTION CLEAN PLOT ===")
    
    sal_clean_plot_path = os.path.join('salary_data2', f'{prefix}_sal_dist_pdf_clean.png')
    if os.path.exists(sal_clean_plot_path):
        print(f"{dataset_name} salary distribution plot already exists at: {sal_clean_plot_path} (skipping creation)")
        return
    
    # Extract and filter salary data
    original_sal = df_orig['Gesamtbetrag_Einkuenfte']
    df_sal_sal = df_sal['Gesamtbetrag_Einkuenfte']
    df_cell_sal_stitched_sal = df_cell_sal_stitched['Gesamtbetrag_Einkuenfte']

    # Filter out values <= 0 for log scale compatibility
    original_sal_filtered = original_sal[original_sal > 0]
    df_sal_sal_filtered = df_sal_sal[df_sal_sal > 0]
    df_cell_sal_stitched_sal_filtered = df_cell_sal_stitched_sal[df_cell_sal_stitched_sal > 0]

    # Function to calculate KDE on log scale with filtered data (> 0)
    def log_kde_filtered(data, x_eval):
        # Perform KDE on log-transformed data
        log_data = np.log10(data)
        kde = gaussian_kde(log_data)
        log_x_eval = np.log10(x_eval)
        
        # Evaluate KDE and transform back to original scale
        # Need to account for the Jacobian of the log transformation
        density = kde(log_x_eval) / (x_eval * np.log(10))
        return density

    print(f"Creating {dataset_name} salary distribution clean plot...")
    fig, ax_main = plt.subplots(figsize=(12, 8))

    # Use filtered data for main plot ranges
    min_val = min(original_sal_filtered.min(), df_sal_sal_filtered.min(), 
                  df_cell_sal_stitched_sal_filtered.min())
    max_val = max(original_sal_filtered.max(), df_sal_sal_filtered.max(), 
                  df_cell_sal_stitched_sal_filtered.max())

    # Create x-axis values for smooth curves (log-spaced for main plot)
    x_smooth = np.logspace(np.log10(min_val), np.log10(max_val), 1000)

    # Plot curves in the specified order on main plot
    # 1. Original curve first (dashed, thicker)
    original_density = log_kde_filtered(original_sal_filtered, x_smooth)
    ax_main.plot(x_smooth, original_density, label='Original', color='blue', 
                 linewidth=2.5, linestyle='--')

    # 2. df_sal curve (renamed to "Syn, income only")  
    df_sal_density = log_kde_filtered(df_sal_sal_filtered, x_smooth)
    ax_main.plot(x_smooth, df_sal_density, label='Syn, income only', color='red', linewidth=2)

    # 3. df_cell_sal_stitched curve (renamed to "Syn, location and income")
    df_cell_sal_stitched_density = log_kde_filtered(df_cell_sal_stitched_sal_filtered, x_smooth)
    ax_main.plot(x_smooth, df_cell_sal_stitched_density, label='Syn, location and income', 
                 color='orange', linewidth=2)

    # Set x-axis to log scale for main plot
    ax_main.set_xscale('log')

    # Customize the main plot
    ax_main.set_xlabel('Income (zero and negative values removed)', fontsize=12)
    ax_main.set_ylabel('Probability Density', fontsize=12)
    # No title as requested
    ax_main.legend(fontsize=10, loc='upper left')  # Move legend to upper left
    ax_main.grid(True, alpha=0.3)

    # Create embedded subplot (linear scale, all data including negatives and zeros)
    ax_inset = fig.add_axes([0.5, 0.45, 0.45, 0.35])  # [left, bottom, width, height] - moved down a bit
    
    # Use all data (unfiltered) for the inset plot
    min_val_all = min(original_sal.min(), df_sal_sal.min(), df_cell_sal_stitched_sal.min())
    max_val_all = max(original_sal.max(), df_sal_sal.max(), df_cell_sal_stitched_sal.max())
    
    # Create x-axis values for linear scale (all data)
    x_smooth_linear = np.linspace(min_val_all, max_val_all, 1000)
    
    # Function for linear KDE (no log transformation, no filtering)
    def linear_kde_all(data, x_eval):
        kde = gaussian_kde(data)
        return kde(x_eval)
    
    # Plot all data on linear scale in the inset
    original_density_linear = linear_kde_all(original_sal, x_smooth_linear)
    ax_inset.plot(x_smooth_linear, original_density_linear, color='blue', linewidth=2, linestyle='--', alpha=0.8)
    
    df_sal_density_linear = linear_kde_all(df_sal_sal, x_smooth_linear)
    ax_inset.plot(x_smooth_linear, df_sal_density_linear, color='red', linewidth=1.5, alpha=0.8)
    
    df_cell_sal_stitched_density_linear = linear_kde_all(df_cell_sal_stitched_sal, x_smooth_linear)
    ax_inset.plot(x_smooth_linear, df_cell_sal_stitched_density_linear, color='orange', linewidth=1.5, alpha=0.8)
    
    # Set x-axis limits
    ax_inset.set_xlim(-900000, 900000)
    
    # Custom tick formatter for 'k' notation
    def thousands_formatter(x, pos):
        if x == 0:
            return '0'
        elif abs(x) >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    
    ax_inset.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    
    # Customize inset plot
    ax_inset.set_xlabel('Income (truncated at 900k)', fontsize=9)
    ax_inset.set_ylabel('Density', fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(sal_clean_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{dataset_name} salary distribution clean plot saved to: {sal_clean_plot_path}")

# Create sal_dist_pdf_clean plots for both datasets
create_sal_dist_clean_plot(df1, d1_sal, d1_cell_sal_stitched, "Dataset 1", "d1")
create_sal_dist_clean_plot(df2, d2_sal, d2_cell_sal_stitched, "Dataset 2", "d2")

print("\n" + "="*60)
print("=== ANALYSIS COMPLETE ===")
print("All synthetic dataframes created and saved to salary_data2/")
print("Plots generated:")
print("  - count_dist_pdf_clean.png (4 lines from both datasets)")
print("  - d1_sal_dist_pdf_clean.png (Dataset 1 salary distribution)")
print("  - d2_sal_dist_pdf_clean.png (Dataset 2 salary distribution)")
print("="*60)
