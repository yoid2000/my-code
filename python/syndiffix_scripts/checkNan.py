import pandas as pd
import os

inPath = os.path.join('c:\\', 'paul', 'sdnist', 'deids', 'national', 'national_mw22_mt0.075.csv')
# Read the DataFrame from a CSV file
df = pd.read_csv(inPath)

# Count the number of NaN values
nan_count = df['DENSITY'].isna().sum()
print(f'The total number of NaN values in the DataFrame is: {nan_count}')
for val in df['DENSITY']:
    if val < 0:
        print(val)
    if val > 100000000:
        print(val)

df['DENSITY'] = pd.to_numeric(df['DENSITY']).astype(int)
print(df['DENSITY'].head())

