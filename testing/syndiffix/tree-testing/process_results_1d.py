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
        pd.DataFrame: DataFrame with columns for nrows, ncols, data_type, skew, bumps, name, ks_sdx, ks_test, ref, leafs_mode, nuniq
    """
    results_dir = "results/1dim"
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
                        'bumps': result['dataset_params']['cols'][0].get('bumps', None),
                        'name': result['dataset_params']['name'],
                        'ks_sdx': result['ks_sdx'],
                        'ks_test': result['ks_test'],
                        'ref': result['run_params']['range_extend_fraction'],
                        'leafs_mode': result['run_params']['leafs_mode'],
                        'nuniq': result['dataset_params']['cols'][0].get('nuniq', None)
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
    df_1d['abs_error'] = df_1d['ks_sdx'] - df_1d['ks_test']
    
    # Calculate relative difference (avoid division by zero)
    df_1d['rel_error'] = (df_1d['ks_sdx'] - df_1d['ks_test']) / np.maximum(df_1d['ks_test'], df_1d['ks_test'])

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
    
    # Create statistics table
    stats_data = {
        'Statistic': ['Min', 'Max', 'Average', 'Median', 'StdDev'],
        'ks_sdx': [
            df_1d['ks_sdx'].min(),
            df_1d['ks_sdx'].max(),
            df_1d['ks_sdx'].mean(),
            df_1d['ks_sdx'].median(),
            df_1d['ks_sdx'].std()
        ],
        'ks_test': [
            df_1d['ks_test'].min(),
            df_1d['ks_test'].max(),
            df_1d['ks_test'].mean(),
            df_1d['ks_test'].median(),
            df_1d['ks_test'].std()
        ],
        'abs_error': [
            df_1d['abs_error'].min(),
            df_1d['abs_error'].max(),
            df_1d['abs_error'].mean(),
            df_1d['abs_error'].median(),
            df_1d['abs_error'].std()
        ],
        'rel_error': [
            df_1d['rel_error'].min(),
            df_1d['rel_error'].max(),
            df_1d['rel_error'].mean(),
            df_1d['rel_error'].median(),
            df_1d['rel_error'].std()
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    print("\nStatistics Summary:")
    print(stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics by ref groups
    if 'ref' in df_1d.columns:
        print(f"\n=== ks_test Statistics by ref (range_extend_fraction) for data_type: {data_type} ===")
        ref_stats_data = []
        ref_values = sorted(df_1d['ref'].unique())
        for ref_val in ref_values:
            ref_data = df_1d[df_1d['ref'] == ref_val]['ks_test']
            ref_stats_data.append({
                'ref': ref_val,
                'count': len(ref_data),
                'min': ref_data.min(),
                'max': ref_data.max(),
                'average': ref_data.mean(),
                'median': ref_data.median(),
                'stddev': ref_data.std()
            })
        
        ref_stats_df = pd.DataFrame(ref_stats_data)
        print(ref_stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics by leafs_mode groups
    if 'leafs_mode' in df_1d.columns:
        print(f"\n=== ks_test Statistics by leafs_mode for data_type: {data_type} ===")
        leafs_stats_data = []
        leafs_mode_values = sorted(df_1d['leafs_mode'].unique())
        for leafs_mode_val in leafs_mode_values:
            leafs_mode_data = df_1d[df_1d['leafs_mode'] == leafs_mode_val]['ks_test']
            leafs_stats_data.append({
                'leafs_mode': leafs_mode_val,
                'count': len(leafs_mode_data),
                'min': leafs_mode_data.min(),
                'max': leafs_mode_data.max(),
                'average': leafs_mode_data.mean(),
                'median': leafs_mode_data.median(),
                'stddev': leafs_mode_data.std()
            })
        
        leafs_stats_df = pd.DataFrame(leafs_stats_data)
        print(leafs_stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics for abs_error by ref groups
    if 'ref' in df_1d.columns:
        print(f"\n=== abs_error Statistics by ref (range_extend_fraction) for data_type: {data_type} ===")
        ref_abs_stats_data = []
        ref_values = sorted(df_1d['ref'].unique())
        for ref_val in ref_values:
            ref_abs_data = df_1d[df_1d['ref'] == ref_val]['abs_error']
            ref_abs_stats_data.append({
                'ref': ref_val,
                'count': len(ref_abs_data),
                'min': ref_abs_data.min(),
                'max': ref_abs_data.max(),
                'average': ref_abs_data.mean(),
                'median': ref_abs_data.median(),
                'stddev': ref_abs_data.std()
            })
        
        ref_abs_stats_df = pd.DataFrame(ref_abs_stats_data)
        print(ref_abs_stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics for abs_error by leafs_mode groups
    if 'leafs_mode' in df_1d.columns:
        print(f"\n=== abs_error Statistics by leafs_mode for data_type: {data_type} ===")
        leafs_abs_stats_data = []
        leafs_mode_values = sorted(df_1d['leafs_mode'].unique())
        for leafs_mode_val in leafs_mode_values:
            leafs_abs_data = df_1d[df_1d['leafs_mode'] == leafs_mode_val]['abs_error']
            leafs_abs_stats_data.append({
                'leafs_mode': leafs_mode_val,
                'count': len(leafs_abs_data),
                'min': leafs_abs_data.min(),
                'max': leafs_abs_data.max(),
                'average': leafs_abs_data.mean(),
                'median': leafs_abs_data.median(),
                'stddev': leafs_abs_data.std()
            })
        
        leafs_abs_stats_df = pd.DataFrame(leafs_abs_stats_data)
        print(leafs_abs_stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics for ks_test by nrows groups
    if 'nrows' in df_1d.columns:
        print(f"\n=== ks_test Statistics by nrows for data_type: {data_type} ===")
        nrows_stats_data = []
        nrows_values = sorted(df_1d['nrows'].unique())
        for nrows_val in nrows_values:
            nrows_data = df_1d[df_1d['nrows'] == nrows_val]['ks_test']
            nrows_stats_data.append({
                'nrows': nrows_val,
                'count': len(nrows_data),
                'min': nrows_data.min(),
                'max': nrows_data.max(),
                'average': nrows_data.mean(),
                'median': nrows_data.median(),
                'stddev': nrows_data.std()
            })
        
        nrows_stats_df = pd.DataFrame(nrows_stats_data)
        print(nrows_stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics for abs_error by nrows groups
    if 'nrows' in df_1d.columns:
        print(f"\n=== abs_error Statistics by nrows for data_type: {data_type} ===")
        nrows_abs_stats_data = []
        nrows_values = sorted(df_1d['nrows'].unique())
        for nrows_val in nrows_values:
            nrows_abs_data = df_1d[df_1d['nrows'] == nrows_val]['abs_error']
            nrows_abs_stats_data.append({
                'nrows': nrows_val,
                'count': len(nrows_abs_data),
                'min': nrows_abs_data.min(),
                'max': nrows_abs_data.max(),
                'average': nrows_abs_data.mean(),
                'median': nrows_abs_data.median(),
                'stddev': nrows_abs_data.std()
            })
        
        nrows_abs_stats_df = pd.DataFrame(nrows_abs_stats_data)
        print(nrows_abs_stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics for ks_test by nuniq groups (if more than one distinct value)
    if 'nuniq' in df_1d.columns and df_1d['nuniq'].notna().sum() > 0:
        nuniq_unique = df_1d['nuniq'].dropna().unique()
        if len(nuniq_unique) > 1:
            print(f"\n=== ks_test Statistics by nuniq for data_type: {data_type} ===")
            nuniq_stats_data = []
            nuniq_values = sorted(nuniq_unique)
            for nuniq_val in nuniq_values:
                nuniq_data = df_1d[df_1d['nuniq'] == nuniq_val]['ks_test']
                nuniq_stats_data.append({
                    'nuniq': nuniq_val,
                    'count': len(nuniq_data),
                    'min': nuniq_data.min(),
                    'max': nuniq_data.max(),
                    'average': nuniq_data.mean(),
                    'median': nuniq_data.median(),
                    'stddev': nuniq_data.std()
                })
            
            nuniq_stats_df = pd.DataFrame(nuniq_stats_data)
            print(nuniq_stats_df.to_string(index=False, float_format='%.3f'))
    
    # Statistics for abs_error by nuniq groups (if more than one distinct value)
    if 'nuniq' in df_1d.columns and df_1d['nuniq'].notna().sum() > 0:
        nuniq_unique = df_1d['nuniq'].dropna().unique()
        if len(nuniq_unique) > 1:
            print(f"\n=== abs_error Statistics by nuniq for data_type: {data_type} ===")
            nuniq_abs_stats_data = []
            nuniq_values = sorted(nuniq_unique)
            for nuniq_val in nuniq_values:
                nuniq_abs_data = df_1d[df_1d['nuniq'] == nuniq_val]['abs_error']
                nuniq_abs_stats_data.append({
                    'nuniq': nuniq_val,
                    'count': len(nuniq_abs_data),
                    'min': nuniq_abs_data.min(),
                    'max': nuniq_abs_data.max(),
                    'average': nuniq_abs_data.mean(),
                    'median': nuniq_abs_data.median(),
                    'stddev': nuniq_abs_data.std()
                })
            
            nuniq_abs_stats_df = pd.DataFrame(nuniq_abs_stats_data)
            print(nuniq_abs_stats_df.to_string(index=False, float_format='%.3f'))
    
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
    
    # Top 5 highest ks_test values where nrows is 1024
    df_1024 = df_1d[df_1d['nrows'] == 1024]
    if len(df_1024) > 0:
        df_1024_sorted = df_1024.sort_values('ks_test', ascending=False)
        print(f"\n=== Top 5 highest ks_test values where nrows=1024 for data_type: {data_type} ===")
        print(df_1024_sorted.head(5)[['name', 'nrows', 'skew', 'bumps', 'ref', 'leafs_mode', 'ks_sdx', 'ks_test']])
    else:
        print(f"\n=== No data with nrows=1024 for data_type: {data_type} ===")
    
    # Top 10 lowest abs_error values where nrows is 1024
    if len(df_1024) > 0:
        df_1024_abs_error_sorted = df_1024.sort_values('abs_error', ascending=True)
        print(f"\n=== Top 10 lowest abs_error values where nrows=1024 for data_type: {data_type} ===")
        print(df_1024_abs_error_sorted.head(10)[['name', 'nrows', 'skew', 'bumps', 'ref', 'leafs_mode', 'ks_sdx', 'ks_test', 'abs_error']])
        
        # Top 10 highest abs_error values where nrows is 1024
        df_1024_abs_error_sorted_desc = df_1024.sort_values('abs_error', ascending=False)
        print(f"\n=== Top 10 highest abs_error values where nrows=1024 for data_type: {data_type} ===")
        print(df_1024_abs_error_sorted_desc.head(10)[['name', 'nrows', 'skew', 'bumps', 'ref', 'leafs_mode', 'ks_sdx', 'ks_test', 'abs_error']])
    else:
        print(f"\n=== No data with nrows=1024 for abs_error analysis for data_type: {data_type} ===")
    
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

def create_ref_leafs_by_abs_error(df_1d, data_type):
    """
    Creates four subplots showing boxplots of abs_error values.
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
    y_min = df_1d['abs_error'].min()
    y_max = df_1d['abs_error'].max()
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
            mode_data = ref_data[ref_data['leafs_mode'] == leafs_mode_val]['abs_error']
            if len(mode_data) > 0:
                boxplot_data.append(mode_data)
                labels.append(leafs_mode_val)
        
        # Create boxplot
        if boxplot_data:
            ax.boxplot(boxplot_data, tick_labels=labels)
            ax.set_title(f'data_type: {data_type}, ref = {ref_val}')
            ax.set_xlabel('leafs_mode')
            ax.set_ylabel('abs_error')
            ax.set_ylim(y_limits)
            ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(ref_values), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    filename = f"processed/abs_error_box_{data_type}.png"
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
        create_ref_leafs_by_abs_error(df_type, data_type)
    
    return df_1d

if __name__ == "__main__":
    main()
