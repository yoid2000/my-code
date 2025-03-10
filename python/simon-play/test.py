import os
from syndiffix import Synthesizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

plots_path = 'plots'

def save_plot(plt, filename):
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    plt.savefig(os.path.join(plots_path, filename))
    plt.close()

def plot_overlay_2dim(df_orig, df_sdx, type):
    if df_orig.shape[1] < 2 or df_sdx.shape[1] < 2:
        raise ValueError("Both DataFrames must have at least two columns")
    
    if not all(df_orig.columns == df_sdx.columns):
        raise ValueError("Both DataFrames must have the same column names")
    
    x_column = df_orig.columns[0]
    y_column = df_orig.columns[1]
    
    plt.figure(figsize=(10, 4))
    
    # Plot for df_orig
    plt.scatter(df_orig[x_column], df_orig[y_column], s=2, color='blue', label='df_orig')
    
    # Plot for df_sdx
    plt.scatter(df_sdx[x_column], df_sdx[y_column], s=2, color='red', label='df_sdx', alpha=0.4)
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xlim(-10000, 1000000)
    plt.title(f'Scatter Plot of {y_column} vs {x_column} ({type})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(plt, f"scatter_overlay_{type}.png")

def plot_scatter_2dim(df_orig, df_sdx, type):
    if df_orig.shape[1] < 2 or df_sdx.shape[1] < 2:
        raise ValueError("Both DataFrames must have at least two columns")
    
    x_column_orig = df_orig.columns[0]
    y_column_orig = df_orig.columns[1]
    
    x_column_sdx = df_sdx.columns[0]
    y_column_sdx = df_sdx.columns[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot for df_orig
    ax1.scatter(df_orig[x_column_orig], df_orig[y_column_orig], s=2)  # Set point size to 1
    ax1.set_xlabel(x_column_orig)
    ax1.set_ylabel(y_column_orig)
    ax1.set_title(f'Scatter Plot of {x_column_orig} vs {y_column_orig} (df_orig) ({type})')
    ax1.grid(True)
    
    # Plot for df_sdx
    ax2.scatter(df_sdx[x_column_sdx], df_sdx[y_column_sdx], s=2)  # Set point size to 1
    ax2.set_xlabel(x_column_sdx)
    ax2.set_ylabel(y_column_sdx)
    ax2.set_title(f'Scatter Plot of {x_column_sdx} vs {y_column_sdx} (df_sdx) ({type})')
    ax2.grid(True)

    
    # Determine the largest x and y axis ranges and extend by 5%
    x_min = min(df_orig[x_column_orig].min(), df_sdx[x_column_sdx].min())
    x_max = max(df_orig[x_column_orig].max(), df_sdx[x_column_sdx].max())
    y_min = min(df_orig[y_column_orig].min(), df_sdx[y_column_sdx].min())
    y_max = max(df_orig[y_column_orig].max(), df_sdx[y_column_sdx].max())
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_min -= 0.05 * x_range
    x_max += 0.05 * x_range
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range
    
    # Set the same x and y axis ranges for both subplots
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    save_plot(plt, f"scatter_plots_{type}.png")

def plot_one_scatter_2dim(df, filename, type):
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two columns")
    
    x_column = df.columns[0]
    y_column = df.columns[1]
    
    plt.figure(figsize=(10, 4))
    plt.scatter(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter Plot for {filename} ({type})')
    plt.grid(True)
    plt.tight_layout()
    save_plot(plt, f"{filename}_{type}.png")

def plot_1col_dist(df_sdx, df_orig, column, type):
    # Compute the CDF for the specified column in df_sdx
    values_sdx = np.sort(df_sdx[column].dropna())
    cdf_sdx = np.arange(1, len(values_sdx) + 1) / len(values_sdx)
    
    # Compute the CDF for the specified column in df_orig
    values_orig = np.sort(df_orig[column].dropna())
    cdf_orig = np.arange(1, len(values_orig) + 1) / len(values_orig)
    
    # Plot the CDFs
    plt.figure(figsize=(10, 6))
    plt.plot(values_sdx, cdf_sdx, label=f'{column} CDF (df_sdx)')
    plt.plot(values_orig, cdf_orig, label=f'{column} CDF (df_orig)')
    plt.xlabel(column)
    plt.ylabel('CDF')
    plt.title(f'Cumulative Distribution Function of {column}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(plt, f'{column}_cdf.png')

df_orig = pd.read_csv('original.csv')

# check if file 'sdx.csv' exists
if os.path.exists('sdx.csv'):
    df_sdx = pd.read_csv('sdx.csv')
else:
    df_sdx = Synthesizer(df_orig).sample()
    df_sdx.to_csv('sdx.csv', index=False)

if os.path.exists('sdx_norm.csv'):
    df_sdx_norm = pd.read_csv('sdx_norm.csv')
else:
    scaler = MinMaxScaler()
    df_orig_norm = pd.DataFrame(scaler.fit_transform(df_orig), columns=df_orig.columns)
    df_sdx_norm = Synthesizer(df_orig_norm).sample()
    df_sdx_norm = pd.DataFrame(scaler.inverse_transform(df_sdx_norm), columns=df_orig.columns)
    df_sdx_norm.to_csv('sdx_norm.csv', index=False)

print(f"Columns: {df_sdx.columns}")

for column in df_sdx.columns:
    plot_1col_dist(df_sdx, df_orig, column, 'raw')
    plot_1col_dist(df_sdx_norm, df_orig, column, 'normalized')

plot_scatter_2dim(df_orig, df_sdx, 'raw')
plot_scatter_2dim(df_orig, df_sdx_norm, 'normalized')

plot_overlay_2dim(df_orig, df_sdx, 'raw')
plot_overlay_2dim(df_orig, df_sdx_norm, 'normalized')