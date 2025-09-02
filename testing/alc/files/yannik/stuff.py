import pandas as pd
import itertools
import os
from syndiffix import Synthesizer

# Check for CSV file first, then Parquet
csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'orig.csv')
parquet_path = os.path.join(os.path.dirname(__file__), 'inputs', 'orig.parquet')

if os.path.exists(csv_path):
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    
    # Remove unnamed columns from the dataframe
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
    
    # Save as parquet and delete CSV
    df.to_parquet(parquet_path)
    os.remove(csv_path)
    print("Converted CSV to Parquet and removed CSV file.")
else:
    print("Reading Parquet file...")
    df = pd.read_parquet(parquet_path)

# Get all columns except EF742
all_columns = df.columns.tolist()
if 'EF742' not in all_columns:
    print("Error: Column 'EF742' not found in the CSV file")
    exit()

# Generate synthetic dataframe
syn_output_dir = os.path.join(os.path.dirname(__file__), 'inputs', 'synthetic_files')
syn_output_path = os.path.join(syn_output_dir, 'syn.parquet')

if os.path.exists(syn_output_path):
    print("Loading existing synthetic dataframe...")
    df_syn = pd.read_parquet(syn_output_path)
else:
    print("Creating synthetic dataframe...")
    df_syn = Synthesizer(df).sample()
    
    # Save synthetic dataframe
    os.makedirs(syn_output_dir, exist_ok=True)
    df_syn.to_parquet(syn_output_path)

other_columns = [col for col in all_columns if col != 'EF742']

# Generate all combinations of EF742 with pairs of other columns
print("Analyzing combinations of EF742 with pairs of other columns:\n")

for col1, col2 in itertools.combinations(other_columns, 2):
    # Count rows where all three columns (EF742, col1, col2) have non-null values
    non_null_count = df[['EF742', col1, col2]].dropna().shape[0]
    
    print(f"EF742 + {col1} + {col2}: {non_null_count} rows without NULL values")

print(f"\nTotal combinations analyzed: {len(list(itertools.combinations(other_columns, 2)))}")

# Repeat the same analysis for synthetic dataframe
print("\n" + "="*60)
print("Analyzing synthetic dataframe (df_syn):")
print("="*60)

for col1, col2 in itertools.combinations(other_columns, 2):
    # Count rows where all three columns (EF742, col1, col2) have non-null values in synthetic data
    non_null_count_syn = df_syn[['EF742', col1, col2]].dropna().shape[0]
    
    print(f"EF742 + {col1} + {col2}: {non_null_count_syn} rows without NULL values (synthetic)")

print(f"\nTotal combinations analyzed (synthetic): {len(list(itertools.combinations(other_columns, 2)))}")
