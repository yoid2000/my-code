import pandas as pd
import json
import os
from syndiffix import Synthesizer
import itertools

import pandas as pd
import numpy as np

intCols= ['PINCP', 'POVPIP', 'NPF', 'NOC', ]

def count_substrings(df, s):
    for col in df.columns:
        # Find rows in column where 's' is a substring
        mask = df[col].astype(str).str.contains(s)
        return mask.sum()

def replace_substring(df, s):
    for col in df.columns:
        # Find rows in column where 's' is a substring
        mask = df[col].astype(str).str.contains(s)
        n_true = mask.sum()
        
        # If there are no True values in mask, skip this column
        if n_true == 0:
            continue
        print(f"{col}: {n_true} of {len(df)} {int(100*(n_true / len(df)))}%") 
        # Find all values in column that do not contain 's'
        good_values = df.loc[~df[col].astype(str).str.contains(s), col]
        
        # If there are no such values, skip this column
        if good_values.empty:
            continue
        
        # Replace values in masked rows with a random choice from good_values
        df.loc[mask, col] = good_values.sample(mask.sum(), replace=True).values
    return df

def doOneSyn(df, cols):
    if len(cols) > 12:
        fileName = 'all_syn'
    else:
        fileName = '.'.join(cols)
    fileName = state + '_' + fileName + '.csv'
    outPath = os.path.join(outDir, fileName)
    print(outPath)
    if os.path.exists(outPath):
        print(f"Already synthesized {outPath}")
        return
    print("type before synthesis:")
    print(df[cols].dtypes)
    df_syn = Synthesizer(df[cols]).sample()
    # Need to convert these back to strings so that the 'N' conversion works
    print("Syn types before conversion:")
    print(df_syn.dtypes)
    for col in cols:
        if col in intCols:
            df_syn[col] = df_syn[col].astype(str)
    print("Syn types after conversion:")
    print(df_syn.dtypes)
    print(df_syn.head())
    print(f"There are {df_syn.isna().sum().sum()} NaN values")
    df_syn = df_syn.fillna('N')
    df_syn = df_syn.replace('<NA>', 'N')
    print(df_syn.head())
    replace_substring(df_syn, '\*')
    df_syn.to_csv(outPath, index=False)

state = 'texas'
rootPath = os.path.join('c:\\', 'paul', 'sdnist')
inPath = os.path.join(rootPath, 'diverse_communities_data_excerpts', state, 'tx2019.csv')
outDir = os.path.join(rootPath, 'deids', state)
print(rootPath)
df = pd.read_csv(inPath)
df = df.replace('N',None)
columns = list(df.columns)
print('Types before conversion:')
print(df.dtypes)
for col in intCols:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
print('Types after conversion:')
print(df.dtypes)

#doOneSyn(df, ['POVPIP'])
#quit()

with open('columns.json', 'r') as f:
    columnSets = json.load(f)
print(columnSets)
for columnSet in columnSets:
    doOneSyn(df, columnSet)
doOneSyn(df, columns)
# We want every column combined with 'SEX'
for col in columns:
    if col == 'SEX':
        continue
    cols = [col, 'SEX']
    doOneSyn(df, cols)
quit()
# Following are special cases for Inconsistency checks
doOneSyn(df, ['AGEP', 'MSP', 'PINCP', 'PINCP_DECILE', 'EDU', 'DPHY', 'DREM'])
for n_dims in [1,2,3]:
    for comb in itertools.combinations(columns,n_dims):
        cols = sorted(list(comb))
        doOneSyn(df, cols)