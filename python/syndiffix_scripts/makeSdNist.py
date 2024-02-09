import pandas as pd
import json
import os
from syndiffix import Synthesizer
from syndiffix.clustering.strategy import DefaultClustering
import itertools

import pandas as pd
import numpy as np

cluster_params = {
    'max_weight':15,       #15
    'sample_size':2000,        #1000
    'merge_threshold':0.1        #0.1
}
cluster_params = {
    'max_weight':30,       #15
    'sample_size':2000,        #1000
    'merge_threshold':0.05        #0.1
}
cluster_params = {
    'max_weight':22,       #15
    'sample_size':2000,        #1000
    'merge_threshold':0.075        #0.1
}
str2IntCols = ['PINCP', 'POVPIP', 'NPF', 'NOC', ]
catCols = [ 'SEX', 'MSP', 'HISP', 'RAC1P', 'HOUSING_TYPE', 
          'OWN_RENT', 'INDP_CAT', 'EDU', 'PINCP_DECILE',
          'DVET', 'DREM', 'DEYE', 'DEAR', 'DPHY', 
          ]
stateConfig = {
    'TX':{'state':'texas','file':'tx2019.csv'},
    'MA':{'state':'massachusetts','file':'ma2019.csv'},
    'NA':{'state':'national','file':'national2019.csv'},
    }
state = 'NA'

def count_substrings(df, s):
    for col in df.columns:
        # Find rows in column where s is a substring
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
        fileName = f"mw{cluster_params['max_weight']}_mt{cluster_params['merge_threshold']}"
    else:
        fileName = '.'.join(cols)
    fileName = stateConfig[state]['state'] + '_' + fileName + '.csv'
    outPath = os.path.join(outDir, fileName)
    print(outPath)
    if os.path.exists(outPath):
        print(f"Already synthesized {outPath}")
        return
    print("type before synthesis:")
    print(df[cols].dtypes)
    df_syn = Synthesizer(df[cols],
                         clustering=DefaultClustering(
                             max_weight=cluster_params['max_weight'],
                             sample_size=cluster_params['sample_size'],
                             merge_threshold=cluster_params['merge_threshold'],
                         )).sample()
    # Need to convert these back to strings so that the 'N' conversion works
    print("Syn types before conversion:")
    print(df_syn.dtypes)
    for col in cols:
        if col in str2IntCols :
            df_syn[col] = df_syn[col].astype(str)
    print("Syn types after conversion:")
    print(df_syn.dtypes)
    print(df_syn.head())
    print(f"There are {df_syn.isna().sum().sum()} NaN values")
    df_syn = df_syn.fillna('N')
    df_syn = df_syn.replace('<NA>', 'N')
    print(df_syn.head())
    replace_substring(df_syn, '\*')
    nSubStr = count_substrings(df_syn, '\*')
    print(f"Got {nSubStr} substrings")
    df_syn.to_csv(outPath, index=False)

rootPath = os.path.join('c:\\', 'paul', 'sdnist')
inPath = os.path.join(rootPath, 'diverse_communities_data_excerpts', stateConfig[state]['state'], stateConfig[state]['file'])
outDir = os.path.join(rootPath, 'deids', stateConfig[state]['state'])
print(rootPath)
df = pd.read_csv(inPath, low_memory=False)
df = df.replace('N',None)
columns = list(df.columns)
print('Types before conversion:')
print(df.dtypes)
for col in str2IntCols:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
for col in catCols:
    if np.issubdtype(df[col].dtypes, np.number) and df[col].isnull().sum() == 0:
        # This is numeric with no NULL values, so convert it to strings
        # so that syndiffix handles it correctly
            df[col] = df[col].astype(str)
print('Types after conversion:')
print(df.dtypes)

with open('columns.json', 'r') as f:
    columnSets = json.load(f)
print(columnSets)
for columnSet in columnSets:
    doOneSyn(df, columnSet)
doOneSyn(df, columns)
quit()
for n_dims in [1,2,3]:
    for comb in itertools.combinations(columns,n_dims):
        cols = sorted(list(comb))
        doOneSyn(df, cols)