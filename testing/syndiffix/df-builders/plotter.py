import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_pdf(df, file_name=None, point_thresh=10):
    """
    Plot the probability distribution function of a single column DataFrame using seaborn.
    High-frequency point values (>= point_thresh occurrences) are shown as 0-width bars.
    
    Args:
        df: DataFrame with one column
        file_name: Optional string to display in the title
        point_thresh: Threshold for separating point values (default: 10)
        
    Returns:
        plt: matplotlib pyplot object
    """
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one column")
    
    column_name = df.columns[0]

    # If the column dtype is str, set point_thresh to 0
    if df[column_name].dtype == 'object':
        point_thresh = 0
    
    # Find values with count >= point_thresh
    value_counts = df[column_name].value_counts()
    point_values = value_counts[value_counts >= point_thresh].index
    
    # Separate dataframes
    df_point = df[df[column_name].isin(point_values)].copy()
    df_remaining = df[~df[column_name].isin(point_values)].copy()
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left y-axis for counts
    ax1.set_ylabel('Count', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create second y-axis for density (right side)
    ax2 = ax1.twinx()
    
    # Plot PDF on right y-axis (density)
    if not df_remaining.empty:
        sns.histplot(data=df_remaining, x=column_name, stat='density', kde=True, bins=50, ax=ax2)
    
    ax2.set_ylabel('Density', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Get histogram data to show counts on left axis
    if not df_remaining.empty:
        # Get the histogram data from the existing plot
        hist_data = ax2.patches
        if hist_data:
            # Calculate counts for each bin
            bin_counts = []
            for patch in hist_data:
                # Calculate count from density and bin width
                density = patch.get_height()
                bin_width = patch.get_width()
                count = density * bin_width * len(df_remaining)
                bin_counts.append(count)
            
            # Set the left y-axis scale based on histogram counts
            if bin_counts:
                max_hist_count = max(bin_counts)
                ax1.set_ylim(0, max(max_hist_count, max(value_counts[point_values]) if len(point_values) > 0 else max_hist_count))
    
    # Add 0-width bars for point values
    if not df_point.empty:
        if df[column_name].dtype == 'object':
            # For categorical data, sort by count (descending)
            point_counts = df_point[column_name].value_counts()  # Already sorted by count
        else:
            # For numerical data, sort by value
            point_counts = df_point[column_name].value_counts().sort_index()
        
        for value, count in point_counts.items():
            ax1.bar(value, count, width=0, edgecolor='red', linewidth=2, alpha=0.7)
            # Add count labels on the bars
            ax1.text(value, count, str(count), ha='center', va='bottom', color='red', fontweight='bold')
    
    # Create title with optional file name
    title = f'Probability Distribution Function of {column_name}'
    if file_name:
        title += f'\n{file_name}'
    
    plt.title(title)
    ax1.set_xlabel(column_name)
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis limits to always be 0 to 1
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    
    return plt

def plot_scatter(df, file_name=None):
    """
    Create a scatter plot from a 2-column DataFrame.
    
    Args:
        df: DataFrame with exactly two columns
        file_name: Optional string to display in the title
        
    Returns:
        plt: matplotlib pyplot object
    """
    if df.shape[1] != 2:
        raise ValueError("DataFrame must have exactly two columns")
    
    col_x = df.columns[0]
    col_y = df.columns[1]
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df[col_x], df[col_y], alpha=0.6)
    
    # Set labels
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    
    # Create title with optional file name
    title = f'Scatter Plot: {col_x} vs {col_y}'
    if file_name:
        title += f'\n{file_name}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_heat(df, file_name=None):
    """
    Create a heatmap from a 2-column DataFrame.
    
    Args:
        df: DataFrame with exactly two columns
        file_name: Optional string to display in the title
        
    Returns:
        plt: matplotlib pyplot object
    """
    if df.shape[1] != 2:
        raise ValueError("DataFrame must have exactly two columns")
    
    col_x = df.columns[0]
    col_y = df.columns[1]
    
    # Determine binning for x-axis
    if df[col_x].dtype == 'object':
        x_bins = sorted(df[col_x].unique())
        x_bin_labels = x_bins
    else:
        x_bins = 20
        x_bin_labels = None
    
    # Determine binning for y-axis
    if df[col_y].dtype == 'object':
        y_bins = sorted(df[col_y].unique())
        y_bin_labels = y_bins
    else:
        y_bins = 20
        y_bin_labels = None
    
    # Create 2D histogram
    if df[col_x].dtype == 'object' and df[col_y].dtype == 'object':
        # Both categorical - create contingency table
        heatmap_data = pd.crosstab(df[col_y], df[col_x])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='viridis')
        
    elif df[col_x].dtype == 'object':
        # X categorical, Y continuous
        plt.figure(figsize=(12, 8))
        
        # Create bins for continuous y-axis
        y_min, y_max = df[col_y].min(), df[col_y].max()
        y_edges = np.linspace(y_min, y_max, y_bins + 1)
        
        # Create heatmap matrix
        heatmap_matrix = np.zeros((y_bins, len(x_bins)))
        
        for i, x_val in enumerate(x_bins):
            x_data = df[df[col_x] == x_val][col_y]
            counts, _ = np.histogram(x_data, bins=y_edges)
            heatmap_matrix[:, i] = counts[::-1]  # Reverse for proper orientation
        
        sns.heatmap(heatmap_matrix, xticklabels=x_bins, 
                   yticklabels=[f'{y_edges[i]:.2f}-{y_edges[i+1]:.2f}' for i in range(y_bins-1, -1, -1)],
                   annot=True, fmt='.0f', cmap='viridis')
        
    elif df[col_y].dtype == 'object':
        # X continuous, Y categorical
        plt.figure(figsize=(12, 8))
        
        # Create bins for continuous x-axis
        x_min, x_max = df[col_x].min(), df[col_x].max()
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        
        # Create heatmap matrix
        heatmap_matrix = np.zeros((len(y_bins), x_bins))
        
        for i, y_val in enumerate(y_bins):
            y_data = df[df[col_y] == y_val][col_x]
            counts, _ = np.histogram(y_data, bins=x_edges)
            heatmap_matrix[len(y_bins)-1-i, :] = counts  # Reverse for proper orientation
        
        sns.heatmap(heatmap_matrix, 
                   xticklabels=[f'{x_edges[i]:.2f}-{x_edges[i+1]:.2f}' for i in range(x_bins)],
                   yticklabels=y_bins[::-1],
                   annot=True, fmt='.0f', cmap='viridis')
        
    else:
        # Both continuous
        plt.figure(figsize=(12, 8))
        counts, x_edges, y_edges = np.histogram2d(df[col_x], df[col_y], bins=[x_bins, y_bins])
        
        # Create labels for axes
        x_labels = [f'{x_edges[i]:.2f}' for i in range(0, len(x_edges)-1, max(1, len(x_edges)//10))]
        y_labels = [f'{y_edges[i]:.2f}' for i in range(0, len(y_edges)-1, max(1, len(y_edges)//10))]
        
        sns.heatmap(counts.T[::-1], cmap='viridis', annot=False)
    
    # Set labels and title
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    
    # Create title with optional file name
    title = f'Heatmap: {col_x} vs {col_y}'
    if file_name:
        title += f'\n{file_name}'
    
    plt.title(title)
    
    return plt

def plot_bins(df, file_name=None):
    """
    Create a bar plot showing each unique value as a separate bin.
    
    Args:
        df: DataFrame with one column
        file_name: Optional string to display in the title
        
    Returns:
        plt: matplotlib pyplot object
    """
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one column")
    
    column_name = df.columns[0]
    
    # Get value counts
    value_counts = df[column_name].value_counts()
    
    # Sort by index for numerical data, keep sorted by count for categorical
    if df[column_name].dtype == 'object':
        # Sort alphabetically for categorical data
        sorted_counts = value_counts.sort_index()
    else:
        # Sort by value for numerical data
        sorted_counts = value_counts.sort_index()
    
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_counts)), sorted_counts.values, alpha=0.7)
    
    # Set x-axis labels
    plt.xticks(range(len(sorted_counts)), sorted_counts.index, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (value, count) in enumerate(sorted_counts.items()):
        plt.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Set labels and title
    plt.xlabel(column_name)
    plt.ylabel('Count')
    
    # Create title with optional file name
    title = f'Bin Counts for {column_name}'
    if file_name:
        title += f'\n{file_name}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return plt