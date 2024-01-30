import os
import pandas as pd
import re
import itertools
from my_sdx_clone import Synthesizer

maxComb = 3
#rootDir = os.path.join('c:\\', 'paul', 'datasets', 'banking.loans')
rootDir = os.path.join('c:\\', 'paul', 'datasets', 'texas')
origDir = os.path.join(rootDir, 'original')
synDir = os.path.join(rootDir, 'syn')

'''
This takes a csv file, and generates synthetic files for 1col, 2col, 3col,
and full.
'''
def doOneSyn(df, cols):
    if len(cols) > maxComb:
        fileName = 'all_syn'
    else:
        fileName = '.'.join(cols)
    fileName = fileName + '.csv'
    outPath = os.path.join(synDir, fileName)
    if os.path.exists(outPath):
        print(f"Already synthesized {outPath}")
        return
    df_syn = Synthesizer(df[cols]).sample()
    df_syn.to_csv(outPath, index=False)

# Get the original file
fileNames = os.listdir(origDir)
if len(fileNames) > 1:
    print("There should be only one file")
if not fileNames[0].endswith('.csv'):
    print("Must be .csv file")
df = pd.read_csv(os.path.join(origDir, fileNames[0]))

columns = list(df.columns)
doOneSyn(df, columns)
for n_dims in range(1,maxComb+1):
    for comb in itertools.combinations(columns,n_dims):
        cols = sorted(list(comb))
        doOneSyn(df, cols)