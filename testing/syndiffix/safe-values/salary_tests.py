import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde
from syndiffix import Synthesizer
from syndiffix.stitcher import stitch

# Read the salary data CSV file
df = pd.read_csv('salary_data/data.csv')

# Display column names and number of rows
print("=== ORIGINAL DATA ANALYSIS ===")
print("Column names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print(f"\nNumber of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

# Display basic info about the dataframe
print("\nDataframe info:")
print(df.info())

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "="*60)
print("=== SYNTHESIZING DATAFRAMES ===")

# Define parquet file paths
df_cell_path = os.path.join('salary_data', 'cell.parquet')
df_cell_sal_path = os.path.join('salary_data', 'cell_sal.parquet')
df_sal_path = os.path.join('salary_data', 'sal.parquet')
df_cell_sal_stitched_path = os.path.join('salary_data', 'cell_sal_stitched.parquet')
df_cell_cnt_path = os.path.join('salary_data', 'cell_cnt.parquet')

# 1. df_cell - synthesizes column Gitterzelle only with safe-values option
print("\n1. Processing df_cell (Gitterzelle only, safe-values)...")
if os.path.exists(df_cell_path):
    print(f"   Loading existing df_cell from: {df_cell_path}")
    df_cell = pd.read_parquet(df_cell_path)
    print(f"   df_cell loaded with {len(df_cell)} rows")
else:
    print("   Creating new df_cell...")
    df_cell = Synthesizer(df[['Gitterzelle']], value_safe_columns=['Gitterzelle']).sample()
    print(f"   df_cell created with {len(df_cell)} rows")

# 2. df_cell_sal - synthesizes Gitterzelle and Gesamtbetrag_Einkuenfte with safe-values for Gitterzelle
print("\n2. Processing df_cell_sal (Gitterzelle + Gesamtbetrag_Einkuenfte, safe-values for Gitterzelle)...")
if os.path.exists(df_cell_sal_path):
    print(f"   Loading existing df_cell_sal from: {df_cell_sal_path}")
    df_cell_sal = pd.read_parquet(df_cell_sal_path)
    print(f"   df_cell_sal loaded with {len(df_cell_sal)} rows")
else:
    print("   Creating new df_cell_sal...")
    df_cell_sal = Synthesizer(df[['Gitterzelle', 'Gesamtbetrag_Einkuenfte']], value_safe_columns=['Gitterzelle']).sample()
    print(f"   df_cell_sal created with {len(df_cell_sal)} rows")

# 3. df_sal - synthesizes Gesamtbetrag_Einkuenfte only, no safe-values
print("\n3. Processing df_sal (Gesamtbetrag_Einkuenfte only, no safe-values)...")
if os.path.exists(df_sal_path):
    print(f"   Loading existing df_sal from: {df_sal_path}")
    df_sal = pd.read_parquet(df_sal_path)
    print(f"   df_sal loaded with {len(df_sal)} rows")
else:
    print("   Creating new df_sal...")
    df_sal = Synthesizer(df[['Gesamtbetrag_Einkuenfte']]).sample()
    print(f"   df_sal created with {len(df_sal)} rows")

# 4. df_cell_sal_stitched - stitched version
print("\n4. Processing df_cell_sal_stitched (stitched from df_cell and df_cell_sal)...")
if os.path.exists(df_cell_sal_stitched_path):
    print(f"   Loading existing df_cell_sal_stitched from: {df_cell_sal_stitched_path}")
    df_cell_sal_stitched = pd.read_parquet(df_cell_sal_stitched_path)
    print(f"   df_cell_sal_stitched loaded with {len(df_cell_sal_stitched)} rows")
else:
    print("   Creating new df_cell_sal_stitched...")
    df_cell_sal_stitched = stitch(df_left=df_cell, df_right=df_cell_sal, shared=False)
    print(f"   df_cell_sal_stitched created with {len(df_cell_sal_stitched)} rows")

# 5. df_cell_cnt - no safe values
df_cell_cnt_syn = None
print("\n5. Processing df_cell_cnt (cell_count only, no safe-values)...")
if os.path.exists(df_cell_cnt_path):
    print(f"   Loading existing df_cell_cnt from: {df_cell_cnt_path}")
    df_cell_cnt = pd.read_parquet(df_cell_cnt_path)
    print(f"   df_cell_cnt loaded with {len(df_cell_cnt)} rows")
else:
    # Make sure column 'cell_count' exists in df
    if 'cell_count' not in df.columns:
        print("    cell_count column not found, skipping")
    else:
        print("   Creating new df_cell_cnt...")
        df_cell_cnt_syn = Synthesizer(df[['cell_count']]).sample()
        print(f"   df_cell_cnt_syn created with {len(df_cell_cnt_syn)} rows")

print("\n" + "="*60)
print("=== SAVING DATAFRAMES ===")

# Save each dataframe as parquet files in salary_data directory
print("\nSaving synthesized dataframes to parquet files...")

# Save df_cell_cnt_syn (only if it was newly created)
if not os.path.exists(df_cell_cnt_path):
    if df_cell_cnt_syn is not None:
        df_cell_cnt_syn.to_parquet(df_cell_cnt_path, index=False)
        print(f"Saved df_cell_cnt to: {df_cell_cnt_path}")
    else:
        print("    cell_count column not found, skipping")
else:
    print(f"df_cell_cnt already exists at: {df_cell_cnt_path} (skipping save)")

# Save df_cell (only if it was newly created)
if not os.path.exists(df_cell_path):
    df_cell.to_parquet(df_cell_path, index=False)
    print(f"Saved df_cell to: {df_cell_path}")
else:
    print(f"df_cell already exists at: {df_cell_path} (skipping save)")

# Save df_cell_sal (only if it was newly created)
if not os.path.exists(df_cell_sal_path):
    df_cell_sal.to_parquet(df_cell_sal_path, index=False)
    print(f"Saved df_cell_sal to: {df_cell_sal_path}")
else:
    print(f"df_cell_sal already exists at: {df_cell_sal_path} (skipping save)")

# Save df_sal (only if it was newly created)
if not os.path.exists(df_sal_path):
    df_sal.to_parquet(df_sal_path, index=False)
    print(f"Saved df_sal to: {df_sal_path}")
else:
    print(f"df_sal already exists at: {df_sal_path} (skipping save)")

# Save df_cell_sal_stitched (only if it was newly created)
if not os.path.exists(df_cell_sal_stitched_path):
    df_cell_sal_stitched.to_parquet(df_cell_sal_stitched_path, index=False)
    print(f"Saved df_cell_sal_stitched to: {df_cell_sal_stitched_path}")
else:
    print(f"df_cell_sal_stitched already exists at: {df_cell_sal_stitched_path} (skipping save)")

print(f"\nAll dataframes processed successfully!")

print("\n" + "="*60)
print("=== GENERAL STATISTICS ===")

# Display general statistics for each dataframe
def display_dataframe_stats(df_name, df):
    print(f"\n--- {df_name} Statistics ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print(f"Missing values:")
    missing = df.isnull().sum()
    for col in df.columns:
        print(f"  {col}: {missing[col]}")
    
    print(f"Unique values:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()}")
    
    print("Summary statistics:")
    print(df.describe(include='all'))
    
    print(f"\nFirst 3 rows:")
    print(df.head(3))

# Display statistics for original data
display_dataframe_stats("ORIGINAL DATA", df)

# Display statistics for synthesized dataframes
display_dataframe_stats("df_cell", df_cell)
display_dataframe_stats("df_cell_sal", df_cell_sal)
display_dataframe_stats("df_sal", df_sal)
display_dataframe_stats("df_cell_sal_stitched", df_cell_sal_stitched)

print("\n" + "="*60)
print("=== COMPARISON SUMMARY ===")
print(f"Original data:           {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"df_cell:                 {df_cell.shape[0]:,} rows × {df_cell.shape[1]} columns")
print(f"df_cell_sal:             {df_cell_sal.shape[0]:,} rows × {df_cell_sal.shape[1]} columns") 
print(f"df_sal:                  {df_sal.shape[0]:,} rows × {df_sal.shape[1]} columns")
print(f"df_cell_sal_stitched:    {df_cell_sal_stitched.shape[0]:,} rows × {df_cell_sal_stitched.shape[1]} columns")

# Compare unique Gitterzelle values
print(f"\nUnique Gitterzelle values:")
print(f"Original:                {df['Gitterzelle'].nunique():,}")
print(f"df_cell:                 {df_cell['Gitterzelle'].nunique():,}")
print(f"df_cell_sal:             {df_cell_sal['Gitterzelle'].nunique():,}")
print(f"df_cell_sal_stitched:    {df_cell_sal_stitched['Gitterzelle'].nunique():,}")

print("\n" + "="*60)
print("=== CREATING SALARY DISTRIBUTION PLOT ===")

# Extract salary data from each dataframe and remove values <= 0 for log scale
original_sal = df['Gesamtbetrag_Einkuenfte']
df_sal_sal = df_sal['Gesamtbetrag_Einkuenfte']
df_cell_sal_sal = df_cell_sal['Gesamtbetrag_Einkuenfte']
df_cell_sal_stitched_sal = df_cell_sal_stitched['Gesamtbetrag_Einkuenfte']

# Filter out values <= 0 for log scale compatibility
original_sal_filtered = original_sal[original_sal > 0]
df_sal_sal_filtered = df_sal_sal[df_sal_sal > 0]
df_cell_sal_sal_filtered = df_cell_sal_sal[df_cell_sal_sal > 0]
df_cell_sal_stitched_sal_filtered = df_cell_sal_stitched_sal[df_cell_sal_stitched_sal > 0]

# Function to calculate KDE on log scale with filtered data (> 0) - needed for both plots
def log_kde_filtered(data, x_eval):
    # Perform KDE on log-transformed data
    log_data = np.log10(data)
    kde = gaussian_kde(log_data)
    log_x_eval = np.log10(x_eval)
    
    # Evaluate KDE and transform back to original scale
    # Need to account for the Jacobian of the log transformation
    density = kde(log_x_eval) / (x_eval * np.log(10))
    return density

# Check if plot already exists
plot_path = os.path.join('salary_data', 'sal_dist_pdf.png')
if os.path.exists(plot_path):
    print(f"Salary distribution plot already exists at: {plot_path} (skipping creation)")
else:
    print("Creating salary distribution plot...")
    # Create probability distribution plot for Gesamtbetrag_Einkuenfte
    plt.figure(figsize=(12, 8))

    # Calculate the minimum and maximum values across all filtered datasets
    min_val = min(original_sal_filtered.min(), df_sal_sal_filtered.min(), 
                  df_cell_sal_sal_filtered.min(), df_cell_sal_stitched_sal_filtered.min())
    max_val = max(original_sal_filtered.max(), df_sal_sal_filtered.max(), 
                  df_cell_sal_sal_filtered.max(), df_cell_sal_stitched_sal_filtered.max())

    # Create x-axis values for smooth curves (log-spaced)
    x_smooth = np.logspace(np.log10(min_val), np.log10(max_val), 1000)

    # Plot smooth probability density curves using filtered data
    df_sal_density = log_kde_filtered(df_sal_sal_filtered, x_smooth)
    plt.plot(x_smooth, df_sal_density, label='df_sal (no safe-values)', color='red', linewidth=2)
    # Add dot at the end of the curve
    plt.plot(x_smooth[-1], df_sal_density[-1], 'o', color='red', markersize=6)

    df_cell_sal_density = log_kde_filtered(df_cell_sal_sal_filtered, x_smooth)
    plt.plot(x_smooth, df_cell_sal_density, label='df_cell_sal (safe Gitterzelle)', color='green', linewidth=2)
    # Add dot at the end of the curve
    plt.plot(x_smooth[-1], df_cell_sal_density[-1], 'o', color='green', markersize=6)

    df_cell_sal_stitched_density = log_kde_filtered(df_cell_sal_stitched_sal_filtered, x_smooth)
    plt.plot(x_smooth, df_cell_sal_stitched_density, label='df_cell_sal_stitched', color='orange', linewidth=2)
    # Add dot at the end of the curve
    plt.plot(x_smooth[-1], df_cell_sal_stitched_density[-1], 'o', color='orange', markersize=6)

    original_density = log_kde_filtered(original_sal_filtered, x_smooth)
    plt.plot(x_smooth, original_density, label='Original', color='blue', linewidth=2)
    # Add dot at the end of the curve
    plt.plot(x_smooth[-1], original_density[-1], 'o', color='blue', markersize=6)

    # Set x-axis to log scale
    plt.xscale('log')

    # Customize the plot
    plt.xlabel('Gesamtbetrag_Einkuenfte', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Probability Distribution of Gesamtbetrag_Einkuenfte (Log Scale)\nComparison Across Original and Synthesized Data (Values > 0 only)', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original: μ={original_sal_filtered.mean():.0f}, σ={original_sal_filtered.std():.0f}, n={len(original_sal_filtered)}")
    text_info.append(f"df_sal: μ={df_sal_sal_filtered.mean():.0f}, σ={df_sal_sal_filtered.std():.0f}, n={len(df_sal_sal_filtered)}")
    text_info.append(f"df_cell_sal: μ={df_cell_sal_sal_filtered.mean():.0f}, σ={df_cell_sal_sal_filtered.std():.0f}, n={len(df_cell_sal_sal_filtered)}")
    text_info.append(f"df_cell_sal_stitched: μ={df_cell_sal_stitched_sal_filtered.mean():.0f}, σ={df_cell_sal_stitched_sal_filtered.std():.0f}, n={len(df_cell_sal_stitched_sal_filtered)}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Salary distribution plot saved to: {plot_path}")

    # Show basic statistics for verification
    print(f"\nBasic statistics verification (filtered data > 0):")
    print(f"Original Gesamtbetrag_Einkuenfte: mean={original_sal_filtered.mean():.2f}, std={original_sal_filtered.std():.2f}, n={len(original_sal_filtered)}")
    print(f"df_sal Gesamtbetrag_Einkuenfte: mean={df_sal_sal_filtered.mean():.2f}, std={df_sal_sal_filtered.std():.2f}, n={len(df_sal_sal_filtered)}")
    print(f"df_cell_sal Gesamtbetrag_Einkuenfte: mean={df_cell_sal_sal_filtered.mean():.2f}, std={df_cell_sal_sal_filtered.std():.2f}, n={len(df_cell_sal_sal_filtered)}")
    print(f"df_cell_sal_stitched Gesamtbetrag_Einkuenfte: mean={df_cell_sal_stitched_sal_filtered.mean():.2f}, std={df_cell_sal_stitched_sal_filtered.std():.2f}, n={len(df_cell_sal_stitched_sal_filtered)}")

    # Show range information and filtering details
    print(f"\nValue range information:")
    print(f"Original: min={original_sal.min():.2f}, max={original_sal.max():.2f} (filtered: min={original_sal_filtered.min():.2f}, max={original_sal_filtered.max():.2f})")
    print(f"df_sal: min={df_sal_sal.min():.2f}, max={df_sal_sal.max():.2f} (filtered: min={df_sal_sal_filtered.min():.2f}, max={df_sal_sal_filtered.max():.2f})")
    print(f"df_cell_sal: min={df_cell_sal_sal.min():.2f}, max={df_cell_sal_sal.max():.2f} (filtered: min={df_cell_sal_sal_filtered.min():.2f}, max={df_cell_sal_sal_filtered.max():.2f})")
    print(f"df_cell_sal_stitched: min={df_cell_sal_stitched_sal.min():.2f}, max={df_cell_sal_stitched_sal.max():.2f} (filtered: min={df_cell_sal_stitched_sal_filtered.min():.2f}, max={df_cell_sal_stitched_sal_filtered.max():.2f})")

    print(f"\nFiltering information for log scale:")
    print(f"Original: {len(original_sal)} total, {len(original_sal_filtered)} after filtering (removed {len(original_sal) - len(original_sal_filtered)} values ≤ 0)")
    print(f"df_sal: {len(df_sal_sal)} total, {len(df_sal_sal_filtered)} after filtering (removed {len(df_sal_sal) - len(df_sal_sal_filtered)} values ≤ 0)")
    print(f"df_cell_sal: {len(df_cell_sal_sal)} total, {len(df_cell_sal_sal_filtered)} after filtering (removed {len(df_cell_sal_sal) - len(df_cell_sal_sal_filtered)} values ≤ 0)")
    print(f"df_cell_sal_stitched: {len(df_cell_sal_stitched_sal)} total, {len(df_cell_sal_stitched_sal_filtered)} after filtering (removed {len(df_cell_sal_stitched_sal) - len(df_cell_sal_stitched_sal_filtered)} values ≤ 0)")

    plt.show()

# Create the clean salary distribution plot
print("\n" + "="*50)
print("=== CREATING CLEAN SALARY DISTRIBUTION PLOT ===")

sal_clean_plot_path = os.path.join('salary_data', 'sal_dist_pdf_clean.png')
sal_clean_plot_path_pdf = os.path.join('salary_data', 'sal_dist_pdf_clean.pdf')
if os.path.exists(sal_clean_plot_path):
    print(f"Clean salary distribution plot already exists at: {sal_clean_plot_path} (skipping creation)")
else:
    print("Creating clean salary distribution plot...")
    fig, ax_main = plt.subplots(figsize=(7, 5))

    # Use the same data ranges as the original plot (filtered data for main plot)
    min_val = min(original_sal_filtered.min(), df_sal_sal_filtered.min(), 
                  df_cell_sal_stitched_sal_filtered.min())
    max_val = max(original_sal_filtered.max(), df_sal_sal_filtered.max(), 
                  df_cell_sal_stitched_sal_filtered.max())

    # Create x-axis values for smooth curves (log-spaced for main plot)
    x_smooth = np.logspace(np.log10(min_val), np.log10(max_val), 1000)

    # Plot curves in the specified order on main plot
    # 1. Original curve first
    original_density = log_kde_filtered(original_sal_filtered, x_smooth)
    ax_main.plot(x_smooth, original_density, label='Original', color='blue', linewidth=2.5, linestyle='--')

    # 2. df_sal curve (renamed to "Syn, income only")  
    df_sal_density = log_kde_filtered(df_sal_sal_filtered, x_smooth)
    ax_main.plot(x_smooth, df_sal_density, label='Syn, income only', color='red', linewidth=2)

    # 3. df_cell_sal_stitched curve (renamed to "Syn, location and income")
    df_cell_sal_stitched_density = log_kde_filtered(df_cell_sal_stitched_sal_filtered, x_smooth)
    ax_main.plot(x_smooth, df_cell_sal_stitched_density, label='Syn, location and income', color='orange', linewidth=2)

    # Set x-axis to log scale for main plot
    ax_main.set_xscale('log')

    # Customize the main plot
    ax_main.set_xlabel('Income (zero and negative values removed, log scale)', fontsize=12)
    ax_main.set_ylabel('Probability Density', fontsize=12)
    # No title as requested
    ax_main.legend(fontsize=10, loc='upper right')
    ax_main.grid(True, alpha=0.3)

    # Create embedded subplot (linear scale, all data including negatives and zeros)
    ax_inset = fig.add_axes([0.5, 0.4, 0.45, 0.35])  # [left, bottom, width, height] - moved to upper right
    
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
    ax_inset.set_xlim(-900000, 400000)
    
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
    ax_inset.set_xlabel('Income (truncated at 400k)', fontsize=9)
    ax_inset.set_ylabel('Density', fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(True, alpha=0.3)

    # No statistics box as requested

    # Save the plot
    plt.tight_layout()
    plt.savefig(sal_clean_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(sal_clean_plot_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clean salary distribution plot saved to: {sal_clean_plot_path}")

# Create the symmetric log scale salary distribution plot
print("\n" + "="*50)
print("=== CREATING SYMMETRIC LOG SALARY DISTRIBUTION PLOT ===")

sal_clean_all_plot_path = os.path.join('salary_data', 'sal_dist_pdf_clean_all.png')
if os.path.exists(sal_clean_all_plot_path):
    print(f"Symmetric log salary distribution plot already exists at: {sal_clean_all_plot_path} (skipping creation)")
else:
    print("Creating symmetric log salary distribution plot...")
    plt.figure(figsize=(12, 8))

    # Separate data into positive and negative parts
    def separate_data(data):
        pos_data = data[data >= 1]  # Positive integers
        neg_data = data[data <= -1]  # Negative integers
        return pos_data, neg_data
    
    # Get positive and negative parts for all datasets
    orig_pos, orig_neg = separate_data(original_sal)
    sal_pos, sal_neg = separate_data(df_sal_sal)
    stitched_pos, stitched_neg = separate_data(df_cell_sal_stitched_sal)
    
    # Create evaluation points for positive and negative sides
    all_pos = np.concatenate([orig_pos, sal_pos, stitched_pos]) if len(orig_pos) > 0 or len(sal_pos) > 0 or len(stitched_pos) > 0 else np.array([])
    all_neg = np.concatenate([orig_neg, sal_neg, stitched_neg]) if len(orig_neg) > 0 or len(sal_neg) > 0 or len(stitched_neg) > 0 else np.array([])
    
    # Create log-spaced evaluation points
    pos_x = np.array([])
    neg_x = np.array([])
    
    if len(all_pos) > 0:
        pos_min = max(1, all_pos.min())
        pos_max = all_pos.max()
        pos_x = np.logspace(np.log10(pos_min), np.log10(pos_max), 500)
    
    if len(all_neg) > 0:
        neg_min = max(1, abs(all_neg.max()))  # Most negative (largest absolute)
        neg_max = abs(all_neg.min())          # Least negative (smallest absolute)
        neg_x_abs = np.logspace(np.log10(neg_min), np.log10(neg_max), 500)
        neg_x = -neg_x_abs[::-1]  # Make negative and reverse order
    
    # Function for KDE on positive or negative data only
    def kde_single_side(data, x_eval):
        if len(data) == 0 or len(x_eval) == 0:
            return np.zeros_like(x_eval)
        kde = gaussian_kde(data)
        return kde(x_eval)
    
    # Plot positive and negative sides separately
    colors = {'Original': 'blue', 'Syn, income only': 'red', 'Syn, location and income': 'orange'}
    styles = {'Original': (2.5, '--'), 'Syn, income only': (2, '-'), 'Syn, location and income': (2, '-')}
    
    datasets = {
        'Original': (orig_pos, orig_neg),
        'Syn, income only': (sal_pos, sal_neg),
        'Syn, location and income': (stitched_pos, stitched_neg)
    }
    
    # Plot each dataset
    for label, (pos_data, neg_data) in datasets.items():
        color = colors[label]
        linewidth, linestyle = styles[label]
        
        # Plot positive side with log-transformed x-axis
        if len(pos_data) > 0 and len(pos_x) > 0:
            pos_density = kde_single_side(pos_data, pos_x)
            # Transform x-axis to log space for plotting
            pos_x_log = np.log10(pos_x)
            plt.plot(pos_x_log, pos_density, label=label, color=color, 
                    linewidth=linewidth, linestyle=linestyle)
        
        # Plot negative side with log-transformed x-axis (negative log space)
        if len(neg_data) > 0 and len(neg_x) > 0:
            neg_density = kde_single_side(neg_data, neg_x)
            # Transform x-axis to negative log space for plotting
            neg_x_log = -np.log10(np.abs(neg_x))
            plt.plot(neg_x_log, neg_density, color=color, 
                    linewidth=linewidth, linestyle=linestyle)
    
    # Set custom x-axis scaling and labels
    ax = plt.gca()
    
    # Create custom ticks in log-transformed space
    pos_tick_values = []
    pos_tick_labels = []
    neg_tick_values = []
    neg_tick_labels = []
    
    if len(pos_x) > 0:
        # Positive ticks: powers of 10 transformed to log space
        pos_min_log = int(np.floor(np.log10(pos_x.min())))
        pos_max_log = int(np.ceil(np.log10(pos_x.max())))
        for i in range(pos_min_log, pos_max_log + 1):
            tick_val = 10**i
            if pos_x.min() <= tick_val <= pos_x.max():
                pos_tick_values.append(i)  # log-transformed position
                if tick_val >= 1000:
                    pos_tick_labels.append(f'{int(tick_val/1000)}k')
                else:
                    pos_tick_labels.append(f'{int(tick_val)}')
    
    if len(neg_x) > 0:
        # Negative ticks: negative powers of 10 transformed to negative log space
        neg_min_log = int(np.floor(np.log10(abs(neg_x.min()))))
        neg_max_log = int(np.ceil(np.log10(abs(neg_x.max()))))
        for i in range(neg_max_log, neg_min_log + 1):
            tick_val = -(10**i)
            if neg_x.min() <= tick_val <= neg_x.max():
                neg_tick_values.append(-i)  # negative log-transformed position
                if abs(tick_val) >= 1000:
                    neg_tick_labels.append(f'-{int(abs(tick_val)/1000)}k')
                else:
                    neg_tick_labels.append(f'{int(tick_val)}')
    
    # Combine ticks and labels
    all_tick_values = neg_tick_values + pos_tick_values
    all_tick_labels = neg_tick_labels + pos_tick_labels
    
    if all_tick_values:
        ax.set_xticks(all_tick_values)
        ax.set_xticklabels(all_tick_labels)
        
        # Set appropriate x-axis limits
        x_min = min(all_tick_values) - 0.2
        x_max = max(all_tick_values) + 0.2
        ax.set_xlim(x_min, x_max)
    
    # Custom formatter for symmetric log scale
    def symlog_formatter(x, pos):
        if abs(x) < 1:
            return '0'
        elif x > 0:
            if x >= 1000:
                return f'{int(x/1000)}k'
            else:
                return f'{int(x)}'
        else:  # x < 0
            abs_x = abs(x)
            if abs_x >= 1000:
                return f'-{int(abs_x/1000)}k'
            else:
                return f'-{int(abs_x)}'
    
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(symlog_formatter))
    
    # Customize the plot
    plt.xlabel('Income (symmetric log scale)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    # No title as requested
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(sal_clean_all_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Symmetric log salary distribution plot saved to: {sal_clean_all_plot_path}")

print("\n" + "="*60)
print("=== CREATING SALARY DISTRIBUTION CDF PLOT ===")

# Check if CDF plot already exists
cdf_plot_path = os.path.join('salary_data', 'sal_dist_cdf.png')
if os.path.exists(cdf_plot_path):
    print(f"Salary distribution CDF plot already exists at: {cdf_plot_path} (skipping creation)")
else:
    print("Creating salary distribution CDF plot...")
    # Create cumulative distribution plot for Gesamtbetrag_Einkuenfte
    plt.figure(figsize=(12, 8))

    # Sort the filtered data for CDF calculation
    original_sal_sorted = np.sort(original_sal_filtered)
    df_sal_sal_sorted = np.sort(df_sal_sal_filtered)
    df_cell_sal_sal_sorted = np.sort(df_cell_sal_sal_filtered)
    df_cell_sal_stitched_sal_sorted = np.sort(df_cell_sal_stitched_sal_filtered)

    # Calculate empirical CDFs
    n_orig = len(original_sal_sorted)
    n_sal = len(df_sal_sal_sorted)
    n_cell_sal = len(df_cell_sal_sal_sorted)
    n_stitched = len(df_cell_sal_stitched_sal_sorted)

    # Plot empirical CDFs
    plt.plot(df_sal_sal_sorted, np.arange(1, n_sal + 1) / n_sal, 
             label='df_sal (no safe-values)', color='red', linewidth=2)
    plt.plot(df_sal_sal_sorted[-1], 1.0, 'o', color='red', markersize=6)

    plt.plot(df_cell_sal_sal_sorted, np.arange(1, n_cell_sal + 1) / n_cell_sal, 
             label='df_cell_sal (safe Gitterzelle)', color='green', linewidth=2)
    plt.plot(df_cell_sal_sal_sorted[-1], 1.0, 'o', color='green', markersize=6)

    plt.plot(df_cell_sal_stitched_sal_sorted, np.arange(1, n_stitched + 1) / n_stitched, 
             label='df_cell_sal_stitched', color='orange', linewidth=2)
    plt.plot(df_cell_sal_stitched_sal_sorted[-1], 1.0, 'o', color='orange', markersize=6)

    plt.plot(original_sal_sorted, np.arange(1, n_orig + 1) / n_orig, 
             label='Original', color='blue', linewidth=2)
    plt.plot(original_sal_sorted[-1], 1.0, 'o', color='blue', markersize=6)

    # Set x-axis to log scale
    plt.xscale('log')

    # Customize the plot
    plt.xlabel('Gesamtbetrag_Einkuenfte', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution Function of Gesamtbetrag_Einkuenfte (Log Scale)\nComparison Across Original and Synthesized Data (Values > 0 only)', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original: μ={original_sal_filtered.mean():.0f}, σ={original_sal_filtered.std():.0f}, n={len(original_sal_filtered)}")
    text_info.append(f"df_sal: μ={df_sal_sal_filtered.mean():.0f}, σ={df_sal_sal_filtered.std():.0f}, n={len(df_sal_sal_filtered)}")
    text_info.append(f"df_cell_sal: μ={df_cell_sal_sal_filtered.mean():.0f}, σ={df_cell_sal_sal_filtered.std():.0f}, n={len(df_cell_sal_sal_filtered)}")
    text_info.append(f"df_cell_sal_stitched: μ={df_cell_sal_stitched_sal_filtered.mean():.0f}, σ={df_cell_sal_stitched_sal_filtered.std():.0f}, n={len(df_cell_sal_stitched_sal_filtered)}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the CDF plot
    plt.tight_layout()
    plt.savefig(cdf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Salary distribution CDF plot saved to: {cdf_plot_path}")

    plt.show()

print("\n" + "="*60)
print("=== CREATING GITTERZELLE COUNT DISTRIBUTION PLOT ===")

# Count occurrences of each Gitterzelle value for each dataframe
original_counts = df['Gitterzelle'].value_counts()
df_cell_counts = df_cell['Gitterzelle'].value_counts()
df_cell_sal_counts = df_cell_sal['Gitterzelle'].value_counts()
df_cell_sal_stitched_counts = df_cell_sal_stitched['Gitterzelle'].value_counts()

# Determine the range for smooth curves (needed for both plots)
min_count = min(original_counts.min(), df_cell_counts.min(), 
                df_cell_sal_counts.min(), df_cell_sal_stitched_counts.min())
max_count = max(original_counts.max(), df_cell_counts.max(), 
                df_cell_sal_counts.max(), df_cell_sal_stitched_counts.max())

# Create x-axis values for smooth curves (linear spacing for non-log scale)
x_smooth = np.linspace(min_count, max_count, 1000)

# Function to calculate KDE on linear scale (needed for both plots)
def linear_kde(data, x_eval):
    kde = gaussian_kde(data)
    return kde(x_eval)

# Check if count plot already exists
count_plot_path = os.path.join('salary_data', 'count_dist_pdf.png')
if os.path.exists(count_plot_path):
    print(f"Gitterzelle count distribution plot already exists at: {count_plot_path} (skipping creation)")
else:
    print("Creating Gitterzelle count distribution plot...")
    # Create probability distribution plot for Gitterzelle counts
    plt.figure(figsize=(12, 8))

    print(f"Count statistics:")
    print(f"Original Gitterzelle counts: min={original_counts.min()}, max={original_counts.max()}, mean={original_counts.mean():.2f}")
    print(f"df_cell Gitterzelle counts: min={df_cell_counts.min()}, max={df_cell_counts.max()}, mean={df_cell_counts.mean():.2f}")
    print(f"df_cell_sal Gitterzelle counts: min={df_cell_sal_counts.min()}, max={df_cell_sal_counts.max()}, mean={df_cell_sal_counts.mean():.2f}")
    print(f"df_cell_sal_stitched Gitterzelle counts: min={df_cell_sal_stitched_counts.min()}, max={df_cell_sal_stitched_counts.max()}, mean={df_cell_sal_stitched_counts.mean():.2f}")

    # Plot smooth probability density curves
    df_cell_density = linear_kde(df_cell_counts.values, x_smooth)
    plt.plot(x_smooth, df_cell_density, label='df_cell (safe Gitterzelle)', color='red', linewidth=2)
    plt.plot(x_smooth[-1], df_cell_density[-1], 'o', color='red', markersize=6)

    df_cell_sal_density = linear_kde(df_cell_sal_counts.values, x_smooth)
    plt.plot(x_smooth, df_cell_sal_density, label='df_cell_sal (safe Gitterzelle)', color='green', linewidth=2)
    plt.plot(x_smooth[-1], df_cell_sal_density[-1], 'o', color='green', markersize=6)

    df_cell_sal_stitched_density = linear_kde(df_cell_sal_stitched_counts.values, x_smooth)
    plt.plot(x_smooth, df_cell_sal_stitched_density, label='df_cell_sal_stitched', color='orange', linewidth=2)
    plt.plot(x_smooth[-1], df_cell_sal_stitched_density[-1], 'o', color='orange', markersize=6)

    original_density = linear_kde(original_counts.values, x_smooth)
    plt.plot(x_smooth, original_density, label='Original', color='blue', linewidth=2)
    plt.plot(x_smooth[-1], original_density[-1], 'o', color='blue', markersize=6)

    # Set x-axis to log scale
    plt.xscale('log')

    # Customize the plot
    plt.xlabel('Count per Gitterzelle', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Probability Distribution of Row Counts per Gitterzelle (Log Scale)\nComparison Across Original and Synthesized Data', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original: μ={original_counts.mean():.1f}, σ={original_counts.std():.1f}, n={len(original_counts)}")
    text_info.append(f"df_cell: μ={df_cell_counts.mean():.1f}, σ={df_cell_counts.std():.1f}, n={len(df_cell_counts)}")
    text_info.append(f"df_cell_sal: μ={df_cell_sal_counts.mean():.1f}, σ={df_cell_sal_counts.std():.1f}, n={len(df_cell_sal_counts)}")
    text_info.append(f"df_cell_sal_stitched: μ={df_cell_sal_stitched_counts.mean():.1f}, σ={df_cell_sal_stitched_counts.std():.1f}, n={len(df_cell_sal_stitched_counts)}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(count_plot_path, dpi=300, bbox_inches='tight')
    print(f"Gitterzelle count distribution plot saved to: {count_plot_path}")

    plt.show()

# Create the clean count distribution plot
print("\n" + "="*50)
print("=== CREATING CLEAN GITTERZELLE COUNT DISTRIBUTION PLOT ===")

count_clean_plot_path = os.path.join('salary_data', 'count_dist_pdf_clean.png')
count_clean_plot_path_pdf = os.path.join('salary_data', 'count_dist_pdf_clean.pdf')
if os.path.exists(count_clean_plot_path):
    print(f"Clean Gitterzelle count distribution plot already exists at: {count_clean_plot_path} (skipping creation)")
else:
    print("Creating clean Gitterzelle count distribution plot...")
    plt.figure(figsize=(7, 4.5))

    # Use the same x_smooth range as the original plot
    min_count = min(original_counts.min(), df_cell_counts.min(), 
                    df_cell_sal_stitched_counts.min())
    max_count = max(original_counts.max(), df_cell_counts.max(), 
                    df_cell_sal_stitched_counts.max())

    # Find the rightmost point of the original curve for x-axis limit
    original_density = linear_kde(original_counts.values, x_smooth)
    # Find where original density drops to a small fraction of its max
    original_max_density = np.max(original_density)
    significant_indices = np.where(original_density > 0.01 * original_max_density)[0]
    if len(significant_indices) > 0:
        rightmost_x = x_smooth[significant_indices[-1]]
    else:
        rightmost_x = max_count

    # Create x-axis values for smooth curves (linear spacing)
    x_smooth_clean = np.linspace(min_count, rightmost_x * 1.1, 1000)  # Extend slightly beyond

    # Plot curves in the specified order
    # 1. Original curve first (dashed, thicker)
    original_density_clean = linear_kde(original_counts.values, x_smooth_clean)
    plt.plot(x_smooth_clean, original_density_clean, label='Original', color='blue', 
             linewidth=3, linestyle='--')

    # 2. df_cell curve (renamed to "Syn, location only")
    df_cell_density_clean = linear_kde(df_cell_counts.values, x_smooth_clean)
    plt.plot(x_smooth_clean, df_cell_density_clean, label='Syn, location only', 
             color='red', linewidth=2)

    # Set x-axis to log scale
    plt.xscale('log')

    # Set x-axis limit near the rightmost point of the original curve
    plt.xlim(min_count, rightmost_x * 1.05)

    # Customize the plot
    plt.xlabel('Cell Count (log scale)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    # No title as requested
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Use decimal notation for x-ticks (not powers) even with log scale
    def decimal_formatter(x, pos):
        return f'{int(x)}'
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(decimal_formatter))
    
    # Set specific x-axis ticks (combining default log ticks with additional ones)
    plt.xticks([1, 2, 5, 10, 20, 50, 100, 200])

    # Add statistics text box in lower left
    text_info = []
    text_info.append(f"Original: μ={original_counts.mean():.1f}, σ={original_counts.std():.1f}, n={len(original_counts)}")
    text_info.append(f"Syn, location only: μ={df_cell_counts.mean():.1f}, σ={df_cell_counts.std():.1f}, n={len(df_cell_counts)}")

    # Position text box in lower left
    plt.text(0.02, 0.02, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(count_clean_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(count_clean_plot_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clean Gitterzelle count distribution plot saved to: {count_clean_plot_path}")

print("\n" + "="*60)
print("=== CREATING GITTERZELLE COUNT DISTRIBUTION CDF PLOT ===")

# Check if count CDF plot already exists
count_cdf_plot_path = os.path.join('salary_data', 'count_dist_cdf.png')
count_cdf_plot_path_pdf = os.path.join('salary_data', 'count_dist_cdf.pdf')
if os.path.exists(count_cdf_plot_path):
    print(f"Gitterzelle count distribution CDF plot already exists at: {count_cdf_plot_path} (skipping creation)")
else:
    print("Creating Gitterzelle count distribution CDF plot...")
    # Create cumulative distribution plot for Gitterzelle counts
    plt.figure(figsize=(12, 8))

    # Sort the count data for CDF calculation
    original_counts_sorted = np.sort(original_counts.values)
    df_cell_counts_sorted = np.sort(df_cell_counts.values)
    df_cell_sal_counts_sorted = np.sort(df_cell_sal_counts.values)
    df_cell_sal_stitched_counts_sorted = np.sort(df_cell_sal_stitched_counts.values)

    # Calculate empirical CDFs
    n_orig_counts = len(original_counts_sorted)
    n_cell_counts = len(df_cell_counts_sorted)
    n_cell_sal_counts = len(df_cell_sal_counts_sorted)
    n_stitched_counts = len(df_cell_sal_stitched_counts_sorted)

    # Plot empirical CDFs
    plt.plot(df_cell_counts_sorted, np.arange(1, n_cell_counts + 1) / n_cell_counts, 
             label='df_cell (safe Gitterzelle)', color='red', linewidth=2)
    plt.plot(df_cell_counts_sorted[-1], 1.0, 'o', color='red', markersize=6)

    plt.plot(df_cell_sal_counts_sorted, np.arange(1, n_cell_sal_counts + 1) / n_cell_sal_counts, 
             label='df_cell_sal (safe Gitterzelle)', color='green', linewidth=2)
    plt.plot(df_cell_sal_counts_sorted[-1], 1.0, 'o', color='green', markersize=6)

    plt.plot(df_cell_sal_stitched_counts_sorted, np.arange(1, n_stitched_counts + 1) / n_stitched_counts, 
             label='df_cell_sal_stitched', color='orange', linewidth=2)
    plt.plot(df_cell_sal_stitched_counts_sorted[-1], 1.0, 'o', color='orange', markersize=6)

    plt.plot(original_counts_sorted, np.arange(1, n_orig_counts + 1) / n_orig_counts, 
             label='Original', color='blue', linewidth=2)
    plt.plot(original_counts_sorted[-1], 1.0, 'o', color='blue', markersize=6)

    # Set x-axis to log scale
    plt.xscale('log')

    # Customize the plot
    plt.xlabel('Count per Gitterzelle', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution Function of Row Counts per Gitterzelle (Log Scale)\nComparison Across Original and Synthesized Data', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original: μ={original_counts.mean():.1f}, σ={original_counts.std():.1f}, n={len(original_counts)}")
    text_info.append(f"df_cell: μ={df_cell_counts.mean():.1f}, σ={df_cell_counts.std():.1f}, n={len(df_cell_counts)}")
    text_info.append(f"df_cell_sal: μ={df_cell_sal_counts.mean():.1f}, σ={df_cell_sal_counts.std():.1f}, n={len(df_cell_sal_counts)}")
    text_info.append(f"df_cell_sal_stitched: μ={df_cell_sal_stitched_counts.mean():.1f}, σ={df_cell_sal_stitched_counts.std():.1f}, n={len(df_cell_sal_stitched_counts)}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the CDF plot
    plt.tight_layout()
    plt.savefig(count_cdf_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(count_cdf_plot_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Gitterzelle count distribution CDF plot saved to: {count_cdf_plot_path}")

    plt.show()

print("\n" + "="*60)
print("=== CREATING SALARY BIN DIFFERENCE PLOT ===")

# Check if salary bin plot already exists
sal_bin_plot_path = os.path.join('salary_data', 'sal_bin.png')
if os.path.exists(sal_bin_plot_path):
    print(f"Salary bin difference plot already exists at: {sal_bin_plot_path} (skipping creation)")
else:
    print("Creating salary bin difference plot...")
    # Create bin difference plot for Gesamtbetrag_Einkuenfte
    plt.figure(figsize=(12, 8))

    # Create 10 equal-count bins based on original data (using filtered data for consistency)
    n_bins = 10
    original_sal_quantiles = np.percentile(original_sal_filtered, np.linspace(0, 100, n_bins + 1))
    
    # Create bin labels for plotting
    bin_labels = []
    for i in range(n_bins):
        if i == 0:
            bin_labels.append(f"≤{original_sal_quantiles[i+1]:.0f}")
        elif i == n_bins - 1:
            bin_labels.append(f">{original_sal_quantiles[i]:.0f}")
        else:
            bin_labels.append(f"{original_sal_quantiles[i]:.0f}-{original_sal_quantiles[i+1]:.0f}")

    # Count data in each bin for each dataset
    def count_in_bins(data, bin_edges):
        return np.histogram(data, bins=bin_edges)[0]
    
    original_counts_binned = count_in_bins(original_sal_filtered, original_sal_quantiles)
    df_sal_counts_binned = count_in_bins(df_sal_sal_filtered, original_sal_quantiles)
    df_cell_sal_counts_binned = count_in_bins(df_cell_sal_sal_filtered, original_sal_quantiles)
    df_cell_sal_stitched_counts_binned = count_in_bins(df_cell_sal_stitched_sal_filtered, original_sal_quantiles)

    # Calculate differences from original
    df_sal_diff = df_sal_counts_binned - original_counts_binned
    df_cell_sal_diff = df_cell_sal_counts_binned - original_counts_binned
    df_cell_sal_stitched_diff = df_cell_sal_stitched_counts_binned - original_counts_binned

    # Create grouped bar chart
    x = np.arange(n_bins)
    width = 0.25

    plt.bar(x - width, df_sal_diff, width, label='df_sal (no safe-values)', color='red', alpha=0.8)
    plt.bar(x, df_cell_sal_diff, width, label='df_cell_sal (safe Gitterzelle)', color='green', alpha=0.8)
    plt.bar(x + width, df_cell_sal_stitched_diff, width, label='df_cell_sal_stitched', color='orange', alpha=0.8)

    # Add horizontal line at zero for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Customize the plot
    plt.xlabel('Salary Bins', fontsize=12)
    plt.ylabel('Count Difference from Original', fontsize=12)
    plt.title('Salary Distribution Bin Count Differences\nSynthetic Data vs Original Data (10 Equal-Count Bins)', fontsize=14, pad=20)
    plt.xticks(x, bin_labels, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage error information to y-axis ticks
    ax = plt.gca()
    y_ticks = ax.get_yticks()
    y_tick_labels = []
    for tick in y_ticks:
        if tick == 0:
            y_tick_labels.append('0 (0%)')
        else:
            # Calculate average percentage error for this tick value across all bins
            avg_pct_error = abs(tick) / np.mean(original_counts_binned) * 100
            y_tick_labels.append(f'{int(tick)} ({avg_pct_error:.1f}%)')
    ax.set_yticklabels(y_tick_labels)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original bins: {len(original_counts_binned)} bins, ~{original_counts_binned[0]} per bin")
    text_info.append(f"Total differences:")
    text_info.append(f"  df_sal: {np.sum(np.abs(df_sal_diff))}")
    text_info.append(f"  df_cell_sal: {np.sum(np.abs(df_cell_sal_diff))}")
    text_info.append(f"  df_cell_sal_stitched: {np.sum(np.abs(df_cell_sal_stitched_diff))}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(sal_bin_plot_path, dpi=300, bbox_inches='tight')
    print(f"Salary bin difference plot saved to: {sal_bin_plot_path}")

    plt.show()

print("\n" + "="*60)
print("=== CREATING GITTERZELLE COUNT BIN DIFFERENCE PLOT ===")

# Check if count bin plot already exists
count_bin_plot_path = os.path.join('salary_data', 'count_bin.png')
if os.path.exists(count_bin_plot_path):
    print(f"Gitterzelle count bin difference plot already exists at: {count_bin_plot_path} (skipping creation)")
else:
    print("Creating Gitterzelle count bin difference plot...")
    # Create bin difference plot for Gitterzelle counts
    plt.figure(figsize=(12, 8))

    # Define bins and function for this section
    n_bins = 10
    def count_in_bins(data, bin_edges):
        return np.histogram(data, bins=bin_edges)[0]

    # Create 10 equal-count bins based on original count data
    # First, get unique values and their frequencies to understand the data better
    unique_counts, count_frequencies = np.unique(original_counts.values, return_counts=True)
    print(f"Count data range: min={original_counts.min()}, max={original_counts.max()}")
    print(f"Unique count values: {unique_counts[:10]}..." if len(unique_counts) > 10 else f"Unique count values: {unique_counts}")
    print(f"Most common counts: {dict(zip(unique_counts[:5], count_frequencies[:5]))}")
    
    # Create bins that avoid duplicates by using unique quantiles
    original_counts_quantiles = np.percentile(original_counts.values, np.linspace(0, 100, n_bins + 1))
    # Remove duplicate edges and ensure we have meaningful bins
    unique_edges = np.unique(original_counts_quantiles)
    
    # If we have fewer unique edges than desired bins, adjust the number of bins
    if len(unique_edges) < n_bins + 1:
        print(f"Warning: Only {len(unique_edges)} unique bin edges available, adjusting from {n_bins} to {len(unique_edges)-1} bins")
        n_bins = len(unique_edges) - 1
        # Recalculate x positions for the adjusted number of bins
        x = np.arange(n_bins)
    
    bin_edges = unique_edges
    print(f"Final bin edges: {bin_edges}")
    
    # Create bin labels for plotting - now with proper unique bins
    count_bin_labels = []
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        if i == n_bins - 1:
            # Last bin: include upper bound
            count_bin_labels.append(f"[{lower:.0f}, {upper:.0f}]")
        else:
            # All other bins: exclude upper bound
            count_bin_labels.append(f"[{lower:.0f}, {upper:.0f})")

    # Count data in each bin for each dataset using the corrected bin edges
    original_binned_counts = count_in_bins(original_counts.values, bin_edges)
    df_cell_binned_counts = count_in_bins(df_cell_counts.values, bin_edges)
    df_cell_sal_binned_counts = count_in_bins(df_cell_sal_counts.values, bin_edges)
    df_cell_sal_stitched_binned_counts = count_in_bins(df_cell_sal_stitched_counts.values, bin_edges)

    # Calculate differences from original
    df_cell_count_diff = df_cell_binned_counts - original_binned_counts
    df_cell_sal_count_diff = df_cell_sal_binned_counts - original_binned_counts
    df_cell_sal_stitched_count_diff = df_cell_sal_stitched_binned_counts - original_binned_counts

    # Create grouped bar chart
    x = np.arange(n_bins)
    width = 0.25

    plt.bar(x - width, df_cell_count_diff, width, label='df_cell (safe Gitterzelle)', color='red', alpha=0.8)
    plt.bar(x, df_cell_sal_count_diff, width, label='df_cell_sal (safe Gitterzelle)', color='green', alpha=0.8)
    plt.bar(x + width, df_cell_sal_stitched_count_diff, width, label='df_cell_sal_stitched', color='orange', alpha=0.8)

    # Add horizontal line at zero for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Customize the plot
    plt.xlabel('Gitterzelle Count Bins [inclusive, exclusive)', fontsize=12)
    plt.ylabel('Count Difference from Original', fontsize=12)
    plt.title('Gitterzelle Count Distribution Bin Count Differences\nSynthetic Data vs Original Data (10 Equal-Count Bins)', fontsize=14, pad=20)
    plt.xticks(x, count_bin_labels, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage error information to y-axis ticks
    ax = plt.gca()
    y_ticks = ax.get_yticks()
    y_tick_labels = []
    for tick in y_ticks:
        if tick == 0:
            y_tick_labels.append('0 (0%)')
        else:
            # Calculate average percentage error for this tick value across all bins
            avg_pct_error = abs(tick) / np.mean(original_binned_counts) * 100
            y_tick_labels.append(f'{int(tick)} ({avg_pct_error:.1f}%)')
    ax.set_yticklabels(y_tick_labels)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original bins: {len(original_binned_counts)} bins, ~{original_binned_counts[0]} per bin")
    text_info.append(f"Total differences:")
    text_info.append(f"  df_cell: {np.sum(np.abs(df_cell_count_diff))}")
    text_info.append(f"  df_cell_sal: {np.sum(np.abs(df_cell_sal_count_diff))}")
    text_info.append(f"  df_cell_sal_stitched: {np.sum(np.abs(df_cell_sal_stitched_count_diff))}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(count_bin_plot_path, dpi=300, bbox_inches='tight')
    print(f"Gitterzelle count bin difference plot saved to: {count_bin_plot_path}")

    plt.show()

print("\n" + "="*60)
print("=== CREATING GITTERZELLE SALARY SUM BIN DIFFERENCE PLOT ===")

# Check if cell_sal_bin_abs plot already exists
cell_sal_bin_abs_plot_path = os.path.join('salary_data', 'cell_sal_bin_abs.png')
if os.path.exists(cell_sal_bin_abs_plot_path):
    print(f"Gitterzelle salary sum bin absolute difference plot already exists at: {cell_sal_bin_abs_plot_path} (skipping creation)")
else:
    print("Creating Gitterzelle salary sum bin absolute difference plot...")
    # Create bin difference plot for salary sums grouped by Gitterzelle
    plt.figure(figsize=(12, 8))

    # Define bins and function for this section
    n_bins = 10
    
    # Get Gitterzelle value counts from original data and sort by count (ascending for increasing order)
    gitterzelle_counts = df['Gitterzelle'].value_counts().sort_values(ascending=True)
    sorted_gitterzellen = gitterzelle_counts.index.tolist()  # Gitterzelle values sorted by count (lowest first)
    
    # Create bins by dividing the sorted Gitterzelle values into equal groups
    total_unique = len(sorted_gitterzellen)
    bin_size = max(1, total_unique // n_bins)
    
    # Create bin edges by selecting every bin_size-th element
    bin_edges = []
    bin_labels = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        if i == n_bins - 1:
            # Last bin gets all remaining elements
            end_idx = total_unique
            bin_gitterzellen = sorted_gitterzellen[start_idx:]
        else:
            end_idx = min((i + 1) * bin_size, total_unique)
            bin_gitterzellen = sorted_gitterzellen[start_idx:end_idx]
        
        bin_edges.append(bin_gitterzellen)
        
        # Create readable bin labels showing count ranges
        bin_counts = [gitterzelle_counts[g] for g in bin_gitterzellen]
        min_count = min(bin_counts)
        max_count = max(bin_counts)
        
        if min_count == max_count:
            bin_labels.append(f"[{min_count}]")
        else:
            bin_labels.append(f"[{min_count}-{max_count}]")

    # Function to compute salary sums within bins
    def compute_salary_sums_by_gitterzelle_bins(data, gitterzelle_col, salary_col, bin_edges):
        """Compute sum of salaries for each Gitterzelle bin"""
        bin_sums = []
        for bin_gitterzellen in bin_edges:
            # Sum salaries for all Gitterzelle values in this bin
            mask = data[gitterzelle_col].isin(bin_gitterzellen)
            bin_sum = data.loc[mask, salary_col].sum()
            bin_sums.append(bin_sum)
        
        return np.array(bin_sums)
    
    # Compute salary sums for each dataset within bins
    original_sal_sums = compute_salary_sums_by_gitterzelle_bins(df, 'Gitterzelle', 'Gesamtbetrag_Einkuenfte', bin_edges)
    df_cell_sal_sums = compute_salary_sums_by_gitterzelle_bins(df_cell_sal, 'Gitterzelle', 'Gesamtbetrag_Einkuenfte', bin_edges)
    df_cell_sal_stitched_sums = compute_salary_sums_by_gitterzelle_bins(df_cell_sal_stitched, 'Gitterzelle', 'Gesamtbetrag_Einkuenfte', bin_edges)

    # Calculate differences from original
    df_cell_sal_sum_diff = df_cell_sal_sums - original_sal_sums
    df_cell_sal_stitched_sum_diff = df_cell_sal_stitched_sums - original_sal_sums

    # Create grouped bar chart
    x = np.arange(n_bins)
    width = 0.35

    plt.bar(x - width/2, df_cell_sal_sum_diff, width, label='df_cell_sal (safe Gitterzelle)', color='green', alpha=0.8)
    plt.bar(x + width/2, df_cell_sal_stitched_sum_diff, width, label='df_cell_sal_stitched', color='orange', alpha=0.8)

    # Add horizontal line at zero for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Customize the plot
    plt.xlabel('Gitterzelle Bins', fontsize=12)
    plt.ylabel('Salary Sum Difference from Original', fontsize=12)
    plt.title('Salary Sum by Gitterzelle Bin Differences (Absolute)\nSynthetic Data vs Original Data (10 Equal-Count Bins)', fontsize=14, pad=20)
    plt.xticks(x, bin_labels, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage error information to y-axis ticks
    ax = plt.gca()
    y_ticks = ax.get_yticks()
    y_tick_labels = []
    for tick in y_ticks:
        if tick == 0:
            y_tick_labels.append('0 (0%)')
        else:
            # Calculate average percentage error for this tick value across all bins
            avg_pct_error = abs(tick) / np.mean(original_sal_sums) * 100
            y_tick_labels.append(f'{int(tick)} ({avg_pct_error:.1f}%)')
    ax.set_yticklabels(y_tick_labels)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original bins: {len(original_sal_sums)} bins")
    text_info.append(f"Avg salary sum per bin: {np.mean(original_sal_sums):.0f}")
    text_info.append(f"Total absolute differences:")
    text_info.append(f"  df_cell_sal: {np.sum(np.abs(df_cell_sal_sum_diff)):.0f}")
    text_info.append(f"  df_cell_sal_stitched: {np.sum(np.abs(df_cell_sal_stitched_sum_diff)):.0f}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(cell_sal_bin_abs_plot_path, dpi=300, bbox_inches='tight')
    print(f"Gitterzelle salary sum bin absolute difference plot saved to: {cell_sal_bin_abs_plot_path}")

    plt.show()

print("\n" + "="*60)
print("=== CREATING GITTERZELLE SALARY SUM BIN RELATIVE ERROR PLOT ===")

# Check if cell_sal_bin_rel plot already exists
cell_sal_bin_rel_plot_path = os.path.join('salary_data', 'cell_sal_bin_rel.png')
if os.path.exists(cell_sal_bin_rel_plot_path):
    print(f"Gitterzelle salary sum bin relative error plot already exists at: {cell_sal_bin_rel_plot_path} (skipping creation)")
else:
    print("Creating Gitterzelle salary sum bin relative error plot...")
    # Create relative error plot for salary sums grouped by Gitterzelle
    plt.figure(figsize=(12, 8))

    # Define bins and function for this section (duplicate the logic from above)
    n_bins = 10
    
    # Get Gitterzelle value counts from original data and sort by count (ascending for increasing order)
    gitterzelle_counts = df['Gitterzelle'].value_counts().sort_values(ascending=True)
    sorted_gitterzellen = gitterzelle_counts.index.tolist()  # Gitterzelle values sorted by count (lowest first)
    
    # Create bins by dividing the sorted Gitterzelle values into equal groups
    total_unique = len(sorted_gitterzellen)
    bin_size = max(1, total_unique // n_bins)
    
    # Create bin edges by selecting every bin_size-th element
    bin_edges = []
    bin_labels = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        if i == n_bins - 1:
            # Last bin gets all remaining elements
            end_idx = total_unique
            bin_gitterzellen = sorted_gitterzellen[start_idx:]
        else:
            end_idx = min((i + 1) * bin_size, total_unique)
            bin_gitterzellen = sorted_gitterzellen[start_idx:end_idx]
        
        bin_edges.append(bin_gitterzellen)
        
        # Create readable bin labels showing count ranges
        bin_counts = [gitterzelle_counts[g] for g in bin_gitterzellen]
        min_count = min(bin_counts)
        max_count = max(bin_counts)
        
        if min_count == max_count:
            bin_labels.append(f"[{min_count}]")
        else:
            bin_labels.append(f"[{min_count}-{max_count}]")

    # Function to compute salary sums within bins
    def compute_salary_sums_by_gitterzelle_bins(data, gitterzelle_col, salary_col, bin_edges):
        """Compute sum of salaries for each Gitterzelle bin"""
        bin_sums = []
        for bin_gitterzellen in bin_edges:
            # Sum salaries for all Gitterzelle values in this bin
            mask = data[gitterzelle_col].isin(bin_gitterzellen)
            bin_sum = data.loc[mask, salary_col].sum()
            bin_sums.append(bin_sum)
        
        return np.array(bin_sums)
    
    # Compute salary sums for each dataset within bins
    original_sal_sums = compute_salary_sums_by_gitterzelle_bins(df, 'Gitterzelle', 'Gesamtbetrag_Einkuenfte', bin_edges)
    df_cell_sal_sums = compute_salary_sums_by_gitterzelle_bins(df_cell_sal, 'Gitterzelle', 'Gesamtbetrag_Einkuenfte', bin_edges)
    df_cell_sal_stitched_sums = compute_salary_sums_by_gitterzelle_bins(df_cell_sal_stitched, 'Gitterzelle', 'Gesamtbetrag_Einkuenfte', bin_edges)
    
    # Calculate relative errors: (syn - org) / org
    df_cell_sal_rel_error = df_cell_sal_sums / original_sal_sums - 1  # Equivalent to (syn - org) / org
    df_cell_sal_stitched_rel_error = df_cell_sal_stitched_sums / original_sal_sums - 1

    # Create grouped bar chart
    x = np.arange(n_bins)
    width = 0.35

    plt.bar(x - width/2, df_cell_sal_rel_error, width, label='df_cell_sal (safe Gitterzelle)', color='green', alpha=0.8)
    plt.bar(x + width/2, df_cell_sal_stitched_rel_error, width, label='df_cell_sal_stitched', color='orange', alpha=0.8)

    # Add horizontal line at zero for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Customize the plot
    plt.xlabel('Gitterzelle Bins', fontsize=12)
    plt.ylabel('Relative Error (syn-org)/org', fontsize=12)
    plt.title('Salary Sum by Gitterzelle Bin Relative Errors\nSynthetic Data vs Original Data (10 Equal-Count Bins)', fontsize=14, pad=20)
    plt.xticks(x, bin_labels, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis ticks as percentages
    ax = plt.gca()
    y_ticks = ax.get_yticks()
    y_tick_labels = [f'{tick:.1%}' for tick in y_ticks]
    ax.set_yticklabels(y_tick_labels)

    # Add some statistics as text on the plot
    text_info = []
    text_info.append(f"Original bins: {len(original_sal_sums)} bins")
    text_info.append(f"Mean absolute relative errors:")
    text_info.append(f"  df_cell_sal: {np.mean(np.abs(df_cell_sal_rel_error)):.1%}")
    text_info.append(f"  df_cell_sal_stitched: {np.mean(np.abs(df_cell_sal_stitched_rel_error)):.1%}")

    # Add text box with statistics
    plt.text(0.02, 0.98, '\n'.join(text_info), transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8, family='monospace')

    # Save the plot
    plt.tight_layout()
    plt.savefig(cell_sal_bin_rel_plot_path, dpi=300, bbox_inches='tight')
    print(f"Gitterzelle salary sum bin relative error plot saved to: {cell_sal_bin_rel_plot_path}")

    plt.show()

print("\n" + "="*60)
print("=== CREATING PER-CELL ERROR ANALYSIS PLOT ===")

# Check if per_cell_error plot already exists
per_cell_error_plot_path = os.path.join('salary_data', 'per_cell_error.png')
if os.path.exists(per_cell_error_plot_path):
    print(f"Per-cell error analysis plot already exists at: {per_cell_error_plot_path} (skipping creation)")
else:
    print("Creating per-cell error analysis plot...")
    
    # Create per-cell error analysis dataframe
    print("Computing per-cell error metrics...")
    
    # Get Gitterzelle values common to both original and stitched dataframes
    original_gitterzellen = set(df['Gitterzelle'].unique())
    stitched_gitterzellen = set(df_cell_sal_stitched['Gitterzelle'].unique())
    common_gitterzellen = sorted(list(original_gitterzellen.intersection(stitched_gitterzellen)))
    
    print(f"Found {len(common_gitterzellen)} common Gitterzelle values")
    
    # Filter both dataframes to only include common Gitterzelle values
    df_orig_filtered = df[df['Gitterzelle'].isin(common_gitterzellen)].copy()
    df_stitched_filtered = df_cell_sal_stitched[df_cell_sal_stitched['Gitterzelle'].isin(common_gitterzellen)].copy()
    
    print("Computing aggregations using groupby operations...")
    
    # Compute aggregations for original data
    orig_agg = df_orig_filtered.groupby('Gitterzelle')['Gesamtbetrag_Einkuenfte'].agg([
        'count', 'sum', 'mean', 'median'
    ]).reset_index()
    orig_agg.columns = ['Gitterzelle', 'orig_count', 'orig_sum', 'orig_mean', 'orig_median']
    
    # Compute aggregations for stitched data
    stitched_agg = df_stitched_filtered.groupby('Gitterzelle')['Gesamtbetrag_Einkuenfte'].agg([
        'count', 'sum', 'mean', 'median'
    ]).reset_index()
    stitched_agg.columns = ['Gitterzelle', 'stitched_count', 'stitched_sum', 'stitched_mean', 'stitched_median']
    
    # Merge the aggregations
    merged_agg = pd.merge(orig_agg, stitched_agg, on='Gitterzelle', how='inner')
    
    # Compute error metrics
    merged_agg['cnt_err_abs'] = merged_agg['stitched_count'] - merged_agg['orig_count']
    
    # Compute relative errors, handling division by zero
    merged_agg['sum_err_rel'] = np.where(
        merged_agg['orig_sum'] != 0,
        (merged_agg['stitched_sum'] - merged_agg['orig_sum']) / merged_agg['orig_sum'],
        0
    )
    
    merged_agg['avg_err_rel'] = np.where(
        merged_agg['orig_mean'] != 0,
        (merged_agg['stitched_mean'] - merged_agg['orig_mean']) / merged_agg['orig_mean'],
        0
    )
    
    merged_agg['med_err_rel'] = np.where(
        merged_agg['orig_median'] != 0,
        (merged_agg['stitched_median'] - merged_agg['orig_median']) / merged_agg['orig_median'],
        0
    )
    
    # Create the final dataframe with required columns
    df_per_cell_error = merged_agg[['Gitterzelle', 'cnt_err_abs', 'sum_err_rel', 'avg_err_rel', 'med_err_rel', 'orig_count']].copy()
    
    print(f"Created per-cell error dataframe with {len(df_per_cell_error)} rows")
    
    # Debug: Check the range of values for each error metric
    print(f"Debug - Error value ranges:")
    print(f"  cnt_err_abs: min={df_per_cell_error['cnt_err_abs'].min():.0f}, max={df_per_cell_error['cnt_err_abs'].max():.0f}")
    print(f"  sum_err_rel: min={df_per_cell_error['sum_err_rel'].min():.3f}, max={df_per_cell_error['sum_err_rel'].max():.3f}")
    print(f"  avg_err_rel: min={df_per_cell_error['avg_err_rel'].min():.3f}, max={df_per_cell_error['avg_err_rel'].max():.3f}")
    print(f"  med_err_rel: min={df_per_cell_error['med_err_rel'].min():.3f}, max={df_per_cell_error['med_err_rel'].max():.3f}")
    
    # Check for any rows where all relative errors are exactly 0
    zero_rows = df_per_cell_error[(df_per_cell_error['sum_err_rel'] == 0) & 
                                  (df_per_cell_error['avg_err_rel'] == 0) & 
                                  (df_per_cell_error['med_err_rel'] == 0)]
    print(f"Debug - Rows with all relative errors = 0: {len(zero_rows)} out of {len(df_per_cell_error)}")
    
    if len(zero_rows) > 0:
        print(f"Debug - Sample of zero-error rows:")
        print(zero_rows[['Gitterzelle', 'cnt_err_abs', 'sum_err_rel', 'avg_err_rel', 'med_err_rel', 'orig_count']].head())
    
    # Sample some non-zero rows for verification
    non_zero_rows = df_per_cell_error[~((df_per_cell_error['sum_err_rel'] == 0) & 
                                        (df_per_cell_error['avg_err_rel'] == 0) & 
                                        (df_per_cell_error['med_err_rel'] == 0))]
    if len(non_zero_rows) > 0:
        print(f"Debug - Sample of non-zero-error rows:")
        print(non_zero_rows[['Gitterzelle', 'cnt_err_abs', 'sum_err_rel', 'avg_err_rel', 'med_err_rel', 'orig_count']].head())
    
    # Create bins based on original row counts - equal-sized bins without splitting same values
    n_bins = 10
    
    # Get unique count values and how many rows each has
    count_value_counts = df_per_cell_error['orig_count'].value_counts().sort_index()
    unique_counts = count_value_counts.index.values
    rows_per_count = count_value_counts.values
    
    print(f"Debug - Unique count values and their row counts:")
    for count_val, num_rows in zip(unique_counts[:10], rows_per_count[:10]):
        print(f"  Count {count_val}: {num_rows} rows")
    if len(unique_counts) > 10:
        print(f"  ... and {len(unique_counts) - 10} more unique count values")
    
    total_rows = len(df_per_cell_error)
    target_rows_per_bin = total_rows // n_bins
    print(f"Target rows per bin: ~{target_rows_per_bin}")
    
    # Group unique count values into bins while keeping same values together
    bin_edges = []
    bin_labels = []
    current_bin_rows = 0
    current_bin_start_idx = 0
    
    for i, (count_val, num_rows) in enumerate(zip(unique_counts, rows_per_count)):
        # Check if adding this count value would exceed our target (but avoid tiny bins)
        would_exceed = (current_bin_rows + num_rows > target_rows_per_bin * 1.5)
        have_minimum = (current_bin_rows >= target_rows_per_bin * 0.5)
        is_last_value = (i == len(unique_counts) - 1)
        too_many_bins = (len(bin_edges) >= n_bins - 1)
        
        if (would_exceed and have_minimum and not is_last_value and not too_many_bins):
            # Finish current bin before adding this count value
            bin_start_count = unique_counts[current_bin_start_idx]
            bin_end_count = unique_counts[i - 1]
            
            # Get all rows for this bin
            mask = (df_per_cell_error['orig_count'] >= bin_start_count) & (df_per_cell_error['orig_count'] <= bin_end_count)
            bin_data = df_per_cell_error[mask]
            bin_edges.append(bin_data.index.tolist())
            
            # Create label
            if bin_start_count == bin_end_count:
                bin_labels.append(f"= {bin_start_count}")
            else:
                bin_labels.append(f"[{bin_start_count}, {bin_end_count}]")
            
            print(f"Debug - Bin {len(bin_edges)}: {bin_labels[-1]} with {len(bin_data)} rows (target: {target_rows_per_bin})")
            
            # Start new bin
            current_bin_start_idx = i
            current_bin_rows = num_rows
        else:
            # Add this count value to current bin
            current_bin_rows += num_rows
    
    # Handle the last bin (collect all remaining count values)
    if current_bin_start_idx < len(unique_counts):
        bin_start_count = unique_counts[current_bin_start_idx]
        bin_end_count = unique_counts[-1]
        
        # Get all rows for this final bin
        mask = (df_per_cell_error['orig_count'] >= bin_start_count) & (df_per_cell_error['orig_count'] <= bin_end_count)
        bin_data = df_per_cell_error[mask]
        bin_edges.append(bin_data.index.tolist())
        
        # Create label
        if bin_start_count == bin_end_count:
            bin_labels.append(f"= {bin_start_count}")
        else:
            bin_labels.append(f"[{bin_start_count}, {bin_end_count}]")
        
        print(f"Debug - Final bin {len(bin_edges)}: {bin_labels[-1]} with {len(bin_data)} rows (target: {target_rows_per_bin})")
    
    n_bins = len(bin_edges)  # Update to actual number of bins created
    print(f"Created {n_bins} bins with roughly equal row counts, no split values")
    
    # Verify no count value appears in multiple bins
    all_counts_in_bins = []
    for i, bin_indices in enumerate(bin_edges):
        bin_counts = df_per_cell_error.loc[bin_indices, 'orig_count'].unique()
        all_counts_in_bins.extend(bin_counts)
        print(f"Debug - Bin {i+1} ({bin_labels[i]}) contains count values: {sorted(bin_counts)}")
    
    # Check for duplicates
    if len(all_counts_in_bins) != len(set(all_counts_in_bins)):
        print("WARNING: Some count values appear in multiple bins!")
    else:
        print("✓ Verified: No count value appears in multiple bins")
    
    # Create the plot with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = ['cnt_err_abs', 'sum_err_rel', 'avg_err_rel', 'med_err_rel']
    titles = [
        'Count Error (Absolute)\nStitched - Original Row Counts',
        'Sum Error (Relative)\n(Stitched - Original) / Original Salary Sums',
        'Average Error (Relative)\n(Stitched - Original) / Original Salary Averages',
        'Median Error (Relative)\n(Stitched - Original) / Original Salary Medians'
    ]
    
    for metric_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[metric_idx]
        
        # Prepare data for boxplots
        box_data = []
        for bin_indices in bin_edges:
            bin_values = df_per_cell_error.loc[bin_indices, metric].values
            box_data.append(bin_values)
            # Debug: Print range for each bin
            print(f"Debug - {metric} bin {len(box_data)}: min={bin_values.min():.3f}, max={bin_values.max():.3f}, count={len(bin_values)}")
        
        # Create vertical boxplots - ensure outliers are shown
        box_plot = ax.boxplot(box_data, vert=True, patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2),
                             whiskerprops=dict(color='black'),
                             capprops=dict(color='black'),
                             showfliers=True,  # Ensure outliers are shown
                             whis=1.5)  # Standard whisker length
        
        # Customize the subplot
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.set_xlabel('Gitterzelle Count Values', fontsize=10)
        
        # Format y-axis based on metric type
        if metric == 'cnt_err_abs':
            ax.set_ylabel('Count Difference', fontsize=10)
        else:
            ax.set_ylabel('Relative Error', fontsize=10)
            # Set y-axis limits to make negative errors visible (-2.5 to +2.5 in decimal = -250% to +250%)
            ax.set_ylim(-2.5, 2.5)
        
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add some basic statistics as text
        overall_stats = df_per_cell_error[metric]
        if metric == 'cnt_err_abs':
            stats_text = f'Overall: μ={overall_stats.mean():.3f}, σ={overall_stats.std():.3f}'
        else:
            stats_text = f'Overall: μ={overall_stats.mean():.3f}, σ={overall_stats.std():.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
    
    plt.suptitle('Per-Cell Error Analysis: Stitched vs Original Data\nBoxplots by Gitterzelle Count Bins', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    plt.savefig(per_cell_error_plot_path, dpi=300, bbox_inches='tight')
    print(f"Per-cell error analysis plot saved to: {per_cell_error_plot_path}")
    
    # Print summary statistics
    print(f"\nPer-cell error summary statistics:")
    print(f"Count errors (absolute):")
    print(f"  Mean: {df_per_cell_error['cnt_err_abs'].mean():.2f}")
    print(f"  Std:  {df_per_cell_error['cnt_err_abs'].std():.2f}")
    print(f"  Range: [{df_per_cell_error['cnt_err_abs'].min():.0f}, {df_per_cell_error['cnt_err_abs'].max():.0f}]")
    
    print(f"Sum errors (relative):")
    print(f"  Mean: {df_per_cell_error['sum_err_rel'].mean():.1%}")
    print(f"  Std:  {df_per_cell_error['sum_err_rel'].std():.1%}")
    print(f"  Range: [{df_per_cell_error['sum_err_rel'].min():.1%}, {df_per_cell_error['sum_err_rel'].max():.1%}]")
    
    print(f"Average errors (relative):")
    print(f"  Mean: {df_per_cell_error['avg_err_rel'].mean():.1%}")
    print(f"  Std:  {df_per_cell_error['avg_err_rel'].std():.1%}")
    print(f"  Range: [{df_per_cell_error['avg_err_rel'].min():.1%}, {df_per_cell_error['avg_err_rel'].max():.1%}]")
    
    print(f"Median errors (relative):")
    print(f"  Mean: {df_per_cell_error['med_err_rel'].mean():.1%}")
    print(f"  Std:  {df_per_cell_error['med_err_rel'].std():.1%}")
    print(f"  Range: [{df_per_cell_error['med_err_rel'].min():.1%}, {df_per_cell_error['med_err_rel'].max():.1%}]")

    plt.show()
