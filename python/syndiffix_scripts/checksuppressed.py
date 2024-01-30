import os
import pandas as pd
import re
import pprint

pp = pprint.PrettyPrinter(indent=4)


'''
I want to look at different dimensions to see if higher dimensions create
more xxx*yyy type values.
'''

dataDir = os.path.join('c:\\', 'paul', 'datasets', 'texas', 'syn')
colData = {}

# Regex pattern to match strings with 0 or more letters followed by '*' followed by 1 or more numbers
pattern = r'[a-zA-Z]*\*\d+'

suppressData = {}
numDistinct = {}
# Iterate over all files in the directory
for filename in os.listdir(dataDir):
    if filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(os.path.join(dataDir, filename))
        columns = list(df.columns)
        numCols = len(columns)

        # Iterate over each column in the DataFrame
        for column in columns:
            # Count the number of matching strings
            count = df[column].astype(str).str.contains(pattern).sum()
            df_matches = df[column].astype(str)[df[column].astype(str).str.contains(pattern)]
            df_other = df[column].astype(str)[~df[column].astype(str).str.contains(pattern)]

            if count > 0:
                colStr = '.'.join(columns)
                if column not in suppressData:
                    suppressData[column] = {}
                if numCols not in suppressData[column]:
                    suppressData[column][numCols] = [{colStr:{'count':count, 'numDistinct':df_other.nunique()}}]
                else:
                    suppressData[column][numCols].append({colStr:count})
                    suppressData[column][numCols].append({colStr:{'count':count, 'numDistinct':df_other.nunique()}})
                #print(f'File: {filename}, Column: {column}, Count: {count}')
                #print(df_matches.head())
            else:
                numDistinct[column] = df[column].nunique()
for column, stuff in suppressData.items():
    if 1 not in stuff:
        print(f"{column}:")
        print(f"Num unique = {numDistinct[column]}")
        pp.pprint(stuff)
