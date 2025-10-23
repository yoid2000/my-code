import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def gather_results():
    """
    Reads all .json files in the results directory and creates a DataFrame
    with one row per file where ncols == 1.
    
    Returns:
        pd.DataFrame: DataFrame with columns for nrows, ncols, data_type, skew, bumps, name, ks_sdx, ks_test, ref, leafs_mode
    """
    results_dir = "results"
    data = []
    
    # Get all .json files in the results directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                
                # Check if this is a 1D dataset (ncols == 1)
                if result['dataset_params']['ncols'] == 1:
                    # Extract the required fields
                    row = {
                        'nrows': result['dataset_params']['nrows'],
                        'ncols': result['dataset_params']['ncols'],
                        'data_type': result['dataset_params']['cols'][0]['type'],
                        'skew': result['dataset_params']['cols'][0]['skew'],
                        'bumps': result['dataset_params']['cols'][0]['bumps'],
                        'name': result['dataset_params']['name'],
                        'ks_sdx': result['ks_sdx'],
                        'ks_test': result['ks_test'],
                        'ref': result['run_params']['range_extend_fraction'],
                        'leafs_mode': result['run_params']['leafs_mode']
                    }
                    data.append(row)
            
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing file {filename}: {e}")
                continue
    
    # Create DataFrame
    df_1d = pd.DataFrame(data)
    return df_1d

def add_error_columns(df_1d):
    """
    Adds error columns to the DataFrame.
    """
    if df_1d.empty:
        print("No data to analyze")
        return df_1d
    
    # Calculate absolute difference
    df_1d['abs_error'] = np.abs(df_1d['ks_sdx'] - df_1d['ks_test'])
    
    # Calculate relative difference (avoid division by zero)
    df_1d['rel_error'] = np.abs(df_1d['ks_sdx'] - df_1d['ks_test']) / np.maximum(df_1d['ks_test'], 1e-10)

    return df_1d
    
def analyze_1d(df_1d, data_type):
    """
    Analyzes the differences between ks_sdx and ks_test in the 1D dataset DataFrame.
    Adds absolute and relative error columns, sorts by absolute error, and prints statistics.
    
    Args:
        df_1d (pd.DataFrame): DataFrame containing ks_sdx and ks_test columns
        data_type (str): The data type being analyzed
    """
    # Sort by absolute error
    df_1d_sorted = df_1d.sort_values('abs_error', ascending=False).reset_index(drop=True)
    
    print(f"\n=== KS Statistics Analysis for data_type: {data_type} ===")
    
    # Statistics for ks_sdx
    print(f"\nks_sdx Statistics (data_type: {data_type}):")
    print(f"  Min:     {df_1d['ks_sdx'].min():.6f}")
    print(f"  Max:     {df_1d['ks_sdx'].max():.6f}")
    print(f"  Average: {df_1d['ks_sdx'].mean():.6f}")
    print(f"  Median:  {df_1d['ks_sdx'].median():.6f}")
    print(f"  StdDev:  {df_1d['ks_sdx'].std():.6f}")
    
    # Statistics for ks_test
    print(f"\nks_test Statistics (data_type: {data_type}):")
    print(f"  Min:     {df_1d['ks_test'].min():.6f}")
    print(f"  Max:     {df_1d['ks_test'].max():.6f}")
    print(f"  Average: {df_1d['ks_test'].mean():.6f}")
    print(f"  Median:  {df_1d['ks_test'].median():.6f}")
    print(f"  StdDev:  {df_1d['ks_test'].std():.6f}")
    
    print(f"\nAbsolute Difference (|ks_sdx - ks_test|) for data_type: {data_type}:")
    print(f"  Min:     {df_1d['abs_error'].min():.6f}")
    print(f"  Max:     {df_1d['abs_error'].max():.6f}")
    print(f"  Average: {df_1d['abs_error'].mean():.6f}")
    print(f"  Median:  {df_1d['abs_error'].median():.6f}")
    print(f"  StdDev:  {df_1d['abs_error'].std():.6f}")
    
    print(f"\nRelative Difference (|ks_sdx - ks_test| / ks_test) for data_type: {data_type}:")
    print(f"  Min:     {df_1d['rel_error'].min():.6f}")
    print(f"  Max:     {df_1d['rel_error'].max():.6f}")
    print(f"  Average: {df_1d['rel_error'].mean():.6f}")
    print(f"  Median:  {df_1d['rel_error'].median():.6f}")
    print(f"  StdDev:  {df_1d['rel_error'].std():.6f}")
    
    # Statistics by ref groups
    if 'ref' in df_1d.columns:
        print(f"\n=== ks_test Statistics by ref (range_extend_fraction) for data_type: {data_type} ===")
        ref_values = sorted(df_1d['ref'].unique())
        for ref_val in ref_values:
            ref_data = df_1d[df_1d['ref'] == ref_val]['ks_test']
            print(f"\nref = {ref_val} (n={len(ref_data)}):")
            print(f"  Min:     {ref_data.min():.6f}")
            print(f"  Max:     {ref_data.max():.6f}")
            print(f"  Average: {ref_data.mean():.6f}")
            print(f"  Median:  {ref_data.median():.6f}")
            print(f"  StdDev:  {ref_data.std():.6f}")
    
    # Statistics by leafs_mode groups
    if 'leafs_mode' in df_1d.columns:
        print(f"\n=== ks_test Statistics by leafs_mode for data_type: {data_type} ===")
        leafs_mode_values = sorted(df_1d['leafs_mode'].unique())
        for leafs_mode_val in leafs_mode_values:
            leafs_mode_data = df_1d[df_1d['leafs_mode'] == leafs_mode_val]['ks_test']
            print(f"\nleafs_mode = {leafs_mode_val} (n={len(leafs_mode_data)}):")
            print(f"  Min:     {leafs_mode_data.min():.6f}")
            print(f"  Max:     {leafs_mode_data.max():.6f}")
            print(f"  Average: {leafs_mode_data.mean():.6f}")
            print(f"  Median:  {leafs_mode_data.median():.6f}")
            print(f"  StdDev:  {leafs_mode_data.std():.6f}")
    
    # Top and bottom 5 for ks_sdx
    df_ks_sdx_sorted = df_1d.sort_values('ks_sdx', ascending=False)
    print(f"\n=== Top 5 highest ks_sdx values for data_type: {data_type} ===")
    print(df_ks_sdx_sorted.head(5)[['name', 'nrows', 'skew', 'bumps', 'ref', 'leafs_mode', 'ks_sdx', 'ks_test']])
    
    print(f"\n=== Top 5 lowest ks_sdx values for data_type: {data_type} ===")
    print(df_ks_sdx_sorted.tail(5)[['name', 'nrows', 'skew', 'bumps', 'ref', 'leafs_mode', 'ks_sdx', 'ks_test']])
    
    # Top and bottom 5 for ks_test
    df_ks_test_sorted = df_1d.sort_values('ks_test', ascending=False)
    print(f"\n=== Top 5 highest ks_test values for data_type: {data_type} ===")
    print(df_ks_test_sorted.head(5)[['name', 'nrows', 'skew', 'bumps', 'ref', 'leafs_mode', 'ks_sdx', 'ks_test']])
    
    print(f"\n=== Top 5 lowest ks_test values for data_type: {data_type} ===")
    print(df_ks_test_sorted.tail(5)[['name', 'nrows', 'skew', 'bumps', 'ref', 'leafs_mode', 'ks_sdx', 'ks_test']])
    
    print(f"\n=== DataFrame sorted by absolute error (highest first) for data_type: {data_type} ===")
    print(df_1d_sorted)

def create_ref_leafs_by_ks_test(df_1d, data_type):
    """
    Creates four subplots showing boxplots of ks_test values.
    Each subplot represents a different ref value, with boxplots for each leafs_mode.
    
    Args:
        df_1d (pd.DataFrame): DataFrame containing the data
        data_type (str): The data type being analyzed
    """
    if df_1d.empty:
        print(f"No data to plot for data_type: {data_type}")
        return
    
    # Create processed directory if it doesn't exist
    os.makedirs("processed", exist_ok=True)
    
    # Get unique ref values and define leafs_mode order
    ref_values = sorted(df_1d['ref'].unique())
    leafs_mode_order = ['none', 'simple', 'leaf_only']
    
    # Calculate global y-axis limits
    y_min = df_1d['ks_test'].min()
    y_max = df_1d['ks_test'].max()
    y_range = y_max - y_min
    y_padding = y_range * 0.05  # 5% padding
    y_limits = (y_min - y_padding, y_max + y_padding)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ref_val in enumerate(ref_values):
        if i >= 4:  # Only handle up to 4 ref values
            break
            
        ax = axes[i]
        ref_data = df_1d[df_1d['ref'] == ref_val]
        
        # Prepare data for boxplot in specified order
        boxplot_data = []
        labels = []
        
        for leafs_mode_val in leafs_mode_order:
            mode_data = ref_data[ref_data['leafs_mode'] == leafs_mode_val]['ks_test']
            if len(mode_data) > 0:
                boxplot_data.append(mode_data)
                labels.append(leafs_mode_val)
        
        # Create boxplot
        if boxplot_data:
            ax.boxplot(boxplot_data, tick_labels=labels)
            ax.set_title(f'data_type: {data_type}, ref = {ref_val}')
            ax.set_xlabel('leafs_mode')
            ax.set_ylabel('ks_test')
            ax.set_ylim(y_limits)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(ref_values), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    filename = f"processed/ks_test_box_{data_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to {filename}")

def main():
    """Main function that calls gather_results() and displays the results."""
    df_1d = gather_results()
    print(f"Found {len(df_1d)} 1D datasets:")

    df_1d = add_error_columns(df_1d)
    
    # Process each data_type separately
    data_types = df_1d['data_type'].unique()
    for data_type in data_types:
        df_type = df_1d[df_1d['data_type'] == data_type].copy()
        print(f"\n{'='*60}")
        print(f"Processing data_type: {data_type} ({len(df_type)} datasets)")
        print(f"{'='*60}")
        
        # Analyze the differences
        analyze_1d(df_type, data_type)
        
        # Create boxplots
        create_ref_leafs_by_ks_test(df_type, data_type)
    
    return df_1d

if __name__ == "__main__":
    main()
