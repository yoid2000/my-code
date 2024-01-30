import os
import pandas as pd
import re
import itertools
from syndiffix import Synthesizer

rootDir = os.path.join('c:\\', 'paul', 'datasets', 'texas')
origDir = os.path.join(rootDir, 'original')
synDir = os.path.join(rootDir, 'syn')

'''
This is for testing why value suppression takes place at higher dimensions.
'''

# Get the original file
fileNames = os.listdir(origDir)
if len(fileNames) > 1:
    print("There should be only one file")
if not fileNames[0].endswith('.csv'):
    print("Must be .csv file")
df = pd.read_csv(os.path.join(origDir, fileNames[0]))

columns = ['NPF','PUMA','WGTP']
df_test = df[columns]
df_syn = Synthesizer(df_test).sample()
df_syn.to_csv('3dim.csv', index=False)

columns = ['NPF']
df_test = df[columns]
df_syn = Synthesizer(df_test).sample()
df_syn.to_csv('1dim.csv', index=False)

