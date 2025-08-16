import pandas as pd
import os

def add_cell_count():
    """
    Reads salary_data/data.csv, adds a cell_count column based on Gitterzelle frequencies,
    and writes the result back to the same file.
    """
    # Define the file path
    data_path = os.path.join('salary_data', 'data.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} does not exist.")
        return
    
    # Read the CSV file
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if Gitterzelle column exists
    if 'Gitterzelle' not in df.columns:
        print("Error: 'Gitterzelle' column not found in the data.")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Count occurrences of each Gitterzelle value
    gitterzelle_counts = df['Gitterzelle'].value_counts()
    print(f"\nGitterzelle count statistics:")
    print(f"  Unique Gitterzelle values: {len(gitterzelle_counts)}")
    print(f"  Min count: {gitterzelle_counts.min()}")
    print(f"  Max count: {gitterzelle_counts.max()}")
    print(f"  Mean count: {gitterzelle_counts.mean():.2f}")
    
    # Show top 10 most frequent Gitterzelle values
    print(f"\nTop 10 most frequent Gitterzelle values:")
    for gitterzelle, count in gitterzelle_counts.head(10).items():
        print(f"  {gitterzelle}: {count} rows")
    
    # Add the cell_count column by mapping each Gitterzelle to its count
    df['cell_count'] = df['Gitterzelle'].map(gitterzelle_counts)
    
    print(f"\nAdded cell_count column:")
    print(f"  New data shape: {df.shape}")
    print(f"  cell_count range: {df['cell_count'].min()} to {df['cell_count'].max()}")
    
    # Verify the mapping worked correctly
    print(f"\nVerification:")
    print(f"  All cell_count values assigned: {df['cell_count'].notna().all()}")
    print(f"  Sample of Gitterzelle -> cell_count mapping:")
    sample_rows = df[['Gitterzelle', 'cell_count']].drop_duplicates().head(10)
    for _, row in sample_rows.iterrows():
        print(f"    Gitterzelle {row['Gitterzelle']} -> cell_count {row['cell_count']}")
    
    # Write back to the same file
    print(f"\nWriting updated data back to: {data_path}")
    df.to_csv(data_path, index=False)
    
    print("✓ Successfully added cell_count column and saved the file.")

if __name__ == "__main__":
    add_cell_count()
