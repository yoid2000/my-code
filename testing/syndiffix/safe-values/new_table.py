import pandas as pd
import numpy as np

# Read the salary data CSV file
df = pd.read_csv('salary_data/data.csv')

# remove all rows where m != 1
df = df[df['m'] == 1]

# Write df back to salary_data/data.csv
df.to_csv('salary_data/data.csv', index=False)

# Print the number of distinct Gitterzelle values
distinct_gitterzelle = df['Gitterzelle'].nunique()
print(f"Number of distinct Gitterzelle values: {distinct_gitterzelle}")

# Create a copy of df
df1 = df.copy()

# Find all Gitterzelle values that have more than 96 rows and remove them from df1
gitterzelle_counts = df1['Gitterzelle'].value_counts()
large_gitterzelle = gitterzelle_counts[gitterzelle_counts > 96]
num_large_gitterzelle = len(large_gitterzelle)
num_rows_to_remove = large_gitterzelle.sum()

print(f"Found {num_large_gitterzelle} Gitterzelle values with more than 96 rows")
print(f"Total rows to be removed: {num_rows_to_remove}")

if num_large_gitterzelle > 0:
    # Remove all rows with these Gitterzelle values
    df1 = df1[~df1['Gitterzelle'].isin(large_gitterzelle.index)]
    print(f"Removed {num_large_gitterzelle} Gitterzelle values and {num_rows_to_remove} corresponding rows from df1")
    print(f"df1 shape after removal: {df1.shape}")

# Select 4000 distinct Gitterzelle values and remove all rows containing those values
unique_gitterzelle = df1['Gitterzelle'].unique()
print(len(unique_gitterzelle), "distinct Gitterzelle values found")
if len(unique_gitterzelle) >= 4000:
    # Randomly select 4000 distinct Gitterzelle values
    np.random.seed(42)  # For reproducibility
    selected_gitterzelle = np.random.choice(unique_gitterzelle, size=4000, replace=False)
    
    # Remove rows with selected Gitterzelle values
    df1 = df1[~df1['Gitterzelle'].isin(selected_gitterzelle)]
    print(f"Removed {len(selected_gitterzelle)} distinct Gitterzelle values")
    print(f"Remaining rows in df1: {len(df1)}")
else:
    print(f"Warning: Only {len(unique_gitterzelle)} distinct Gitterzelle values available, cannot remove 4000")

# Find rows where Gesamtbetrag_Einkuenfte is 0
zero_mask = df1['Gesamtbetrag_Einkuenfte'] == 0
zero_indices = df1.index[zero_mask]
num_zeros = len(zero_indices)

print(f"Number of rows with Gesamtbetrag_Einkuenfte = 0: {num_zeros}")

if num_zeros > 0:
    # Change 5% of the 0 values to 1
    num_to_change_to_1 = int(np.ceil(num_zeros * 0.05))
    indices_to_1 = np.random.choice(zero_indices, size=min(num_to_change_to_1, num_zeros), replace=False)
    df1.loc[indices_to_1, 'Gesamtbetrag_Einkuenfte'] = 1
    print(f"Changed {len(indices_to_1)} zero values to 1")
    
    # Update zero indices after first change
    remaining_zero_mask = df1['Gesamtbetrag_Einkuenfte'] == 0
    remaining_zero_indices = df1.index[remaining_zero_mask]
    
    # Change 2% of the original 0 values to -1
    num_to_change_to_neg1 = int(np.ceil(num_zeros * 0.02))
    if len(remaining_zero_indices) >= num_to_change_to_neg1:
        indices_to_neg1 = np.random.choice(remaining_zero_indices, size=num_to_change_to_neg1, replace=False)
        df1.loc[indices_to_neg1, 'Gesamtbetrag_Einkuenfte'] = -1
        print(f"Changed {len(indices_to_neg1)} zero values to -1")
    else:
        print(f"Warning: Only {len(remaining_zero_indices)} zero values remaining, changing all to -1")
        df1.loc[remaining_zero_indices, 'Gesamtbetrag_Einkuenfte'] = -1

# Apply the transformation: new_v = int(round(v + (v*0.15)))
df1['Gesamtbetrag_Einkuenfte'] = df1['Gesamtbetrag_Einkuenfte'].apply(
    lambda v: int(round(v + (v * 0.15)))
)

# Save df1 as salary_data/data1.csv
df1.to_csv('salary_data/data1.csv', index=False)
print("Saved df1 as salary_data/data1.csv")

# Display statistics of df and df1
print("\n" + "="*50)
print("STATISTICS COMPARISON")
print("="*50)

print("\nOriginal DataFrame (df) statistics:")
print(f"Shape: {df.shape}")
print(f"Distinct Gitterzelle values: {df['Gitterzelle'].nunique()}")
print("\nGesamtbetrag_Einkuenfte statistics:")
print(df['Gesamtbetrag_Einkuenfte'].describe())

print("\nModified DataFrame (df1) statistics:")
print(f"Shape: {df1.shape}")
print(f"Distinct Gitterzelle values: {df1['Gitterzelle'].nunique()}")
print("\nGesamtbetrag_Einkuenfte statistics:")
print(df1['Gesamtbetrag_Einkuenfte'].describe())

print("\n" + "="*50)
print("VALUE COUNTS COMPARISON")
print("="*50)

print("\nOriginal df - Gesamtbetrag_Einkuenfte value counts (top 10):")
print(df['Gesamtbetrag_Einkuenfte'].value_counts().head(10))

print("\nModified df1 - Gesamtbetrag_Einkuenfte value counts (top 10):")
print(df1['Gesamtbetrag_Einkuenfte'].value_counts().head(10))