import os
import pandas as pd
import matplotlib.pyplot as plt

def get_stats(parquet_path, blob_path):

    max_rows = 250000
    max_columns = 50
    stats = []
    # Iterate over each file in the directory
    for filename in os.listdir(parquet_path):
        if filename.endswith('.parquet'):
            file_path = os.path.join(parquet_path, filename)
            
            # Read the .parquet file as a dataframe
            df = pd.read_parquet(file_path)
            
            # Get the number of rows and columns
            num_rows = len(df)
            num_columns = len(df.columns)
            # if the number of rows or columns exceeds the maximum, remove this file
            if num_rows > max_rows or num_columns > max_columns:
                os.remove(file_path)
                continue
            
            # Add the stats to the array
            stats.append((num_rows, num_columns))
    
    # Convert the stats to a DataFrame for easier plotting
    stats_df = pd.DataFrame(stats, columns=['Rows', 'Columns'])
    
    # Plot the stats as a scatterplot
    plt.scatter(stats_df['Rows'], stats_df['Columns'])
    plt.xlabel('Rows')
    plt.ylabel('Columns')
    # set x axis limits
    plt.title('Scatterplot of Rows vs Columns in Parquet Files')
    plt.tight_layout()
    plt.savefig(os.path.join(blob_path, 'rows_cols.png'))
    plt.close()
    
    # Plot the stats as a scatterplot, log scale
    plt.scatter(stats_df['Rows'], stats_df['Columns'])
    plt.xlabel('Rows')
    plt.ylabel('Columns')
    plt.title('Scatterplot of Rows vs Columns in Parquet Files')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(blob_path, 'rows_cols_log.png'))
    plt.close()

blob_code_path = '.'
# check if there is an environment variable called BLOB_TEST_DIR
if 'BLOB_TEST_DIR' in os.environ:
    blob_path = os.environ['BLOB_TEST_DIR']
    blob_code_path = os.environ['BLOB_TEST_CODE']
else:
    blob_path = os.path.join(os.getcwd(), 'blob_tests')

print(f"Blob path: {blob_path}")

# make directory 'datasets' if it doesn't exist
datasets_path = os.path.join(blob_path, 'datasets')
kaggle_parquet_path = os.path.join(datasets_path, 'kaggle_parquet')

# Example usage
get_stats(kaggle_parquet_path, blob_path)