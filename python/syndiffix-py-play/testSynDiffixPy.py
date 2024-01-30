import os
import sys
import time
import json
import datetime as dt
import s2stuff

import pandas as pd
import psutil
from math import cos, asin, sqrt, pi
sys.path.insert(0, os.path.join('c:\\', 'paul', 'GitHub', 'syndiffix-py'))

from syndiffix.synthesizer import Synthesizer

numRows = 10000

# Utility function for loading a CSV file.
def load_csv(path: str) -> pd.DataFrame:
    from pandas.errors import ParserError

    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)

    # Try to infer datetime columns.
    for col in df.columns[df.dtypes == "object"]:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ParserError, ValueError):
            pass

    return df

# Distance between two x,y points, on a flat surface
def xyDist(x1, y1, x2, y2):
    return sqrt(pow(x2-x1,2) + pow(y2-y1,2))

# This version takes into account curvature of earth. Overkill for what we're doing here...
def latLonDist(lat1, lon1, lat2, lon2):
    r = 6371 # km
    p = pi / 180

    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 2 * r * asin(sqrt(a))

class testSynDiffixPy:
    def __init__(self, df, table, columns, outDir, aidsColumns=None, force=False):
        self.outStats = None
        self.dfAnon = None
        self.df = df
        self.table = table
        self.columns = columns
        self.outDir = outDir

        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory_usage = process.memory_info().rss

        print("\nFitting the synthesizer over the data...")
        aids = None
        if len(columns) > 0:
            self.dfOrig = self.df[columns]
            self.outFileRoot = table + '_' + '_'.join(columns)
        else:
            self.dfOrig = self.df
            self.outFileRoot = table + '_all'
        if aidsColumns:
            self.outFileRoot += '.aids.' + '_'.join(aidsColumns)
            aids = self.df[aidsColumns]
        self.csvPath = os.path.join(self.outDir, self.outFileRoot+'.csv')
        if force is False and os.path.exists(self.csvPath):
            print(f"Already completed: {self.csvPath}")
            return
        print(f"Build '{self.csvPath}'")
        synthesizer = Synthesizer(self.dfOrig, aids=aids)
        
        syn_time = time.time()

        print("Column clusters:")
        print("Initial=", synthesizer.clusters.initial_cluster)
        for cluster in synthesizer.clusters.derived_clusters:
            print("Derived=", cluster)

        print("\nSampling rows from the synthesizer...")
        self.dfAnon = synthesizer.sample()

        end_time = time.time()
        syn_elapsed = syn_time - start_time
        sample_elapsed = end_time - syn_time
        elapsed = end_time - start_time

        memory_usage = (process.memory_info().rss - start_memory_usage) // (1024**2)
        print(f"Runtime: {round(elapsed)} seconds. Memory usage: {memory_usage} MB.")

        self.outStats = {'table':table,
                    'columns':columns,
                    'numRowsOrig':self.dfOrig.shape[0],
                    'numRowsAnon':self.dfAnon.shape[0],
                    'typesOrig': [str(x) for x in self.dfOrig.dtypes],
                    'typesAnon': [str(x) for x in self.dfAnon.dtypes],
                    'syn_elapsed':syn_elapsed,
                    'sample_elapsed':sample_elapsed,
                    'all_elapsed':elapsed,
                    'init_sub_table':synthesizer.clusters.initial_cluster,
                    'derived_sub_tables': synthesizer.clusters.derived_clusters,
                    'memory_usage': memory_usage,
                    'more': {},
                    }
        self.moreStats = self.outStats['more']

        print(f"\nWriting sampled rows to `{self.csvPath}`...")
        self.dfAnon.to_csv(self.csvPath, index=False)
        self.saveStats()

    def saveStats(self):
        if self.outStats:
            statsPath = os.path.join(self.outDir, self.outFileRoot+'.stats.json')
            with open(statsPath, 'w') as f:
                json.dump(self.outStats, f, indent=4)

# Simple usage example of the SynDiffix library.
# This script assumes each row belongs to a different protected entity.
# All columns in the input file are processed.
output_dir = os.path.join('c:\\', 'paul', 'abData', 'sdx_py_tests')
os.makedirs(output_dir, exist_ok=True)

csvFile = 'taxi-one-day.csv'
csvPath = os.path.join('c:\\', 'paul', 'abData', 'csvAb', 'original', csvFile)
aidsColumns = ['hack']
#aidsColumns = None

print(f"Loading data from `{csvPath}`...")
dfAll = load_csv(csvPath)
dfAll['dropoff_longitude'] = pd.to_numeric(dfAll['dropoff_longitude'], errors='coerce')
dfAll['dropoff_latitude'] = pd.to_numeric(dfAll['dropoff_latitude'], errors='coerce')

print(f"Loaded {len(dfAll)} rows. Columns:")
for i, (column, dtype) in enumerate(zip(dfAll.columns, dfAll.dtypes)):
    print(f"{i}: {column} ({dtype})")
df = dfAll.sample(n=numRows, random_state=1)
print(f"Sampled file has {len(df)} rows")

# Test number of unique hacks (without AID column)
tsd = testSynDiffixPy(df, csvFile, ['med','hack'], output_dir)
if tsd.dfAnon is not None:
    tsd.moreStats['uniqueHacks'] = {'orig': tsd.dfOrig['hack'].nunique(),
                                    'anon': tsd.dfAnon['hack'].nunique()}
    tsd.saveStats()

# Same test, but with AID columns
tsd = testSynDiffixPy(df, csvFile, ['med','hack'], output_dir, aidsColumns=aidsColumns, force=False)
if tsd.dfAnon is not None:
    tsd.moreStats['uniqueHacks'] = {'orig': tsd.dfOrig['hack'].nunique(),
                                    'anon': tsd.dfAnon['hack'].nunique()}
    tsd.saveStats()

# Test relationship between pickup and dropoff datetimes
tsd = testSynDiffixPy(df, csvFile, ['pickup_datetime', 'dropoff_datetime'], output_dir, aidsColumns=aidsColumns, force=False)
if tsd.dfAnon is not None:
    tsd.dfOrig['diff'] = tsd.dfOrig.apply(lambda x: (x['dropoff_datetime'] - x['pickup_datetime']).total_seconds(), axis=1)
    tsd.dfAnon['diff'] = tsd.dfAnon.apply(lambda x: (x['dropoff_datetime'] - x['pickup_datetime']).total_seconds(), axis=1)
    tsd.moreStats['tripTime'] = {
        'mean': {'orig': tsd.dfOrig.loc[:, 'diff'].mean(),
                 'anon': tsd.dfAnon.loc[:, 'diff'].mean(),
                 },
        'std': {'orig': tsd.dfOrig.loc[:, 'diff'].std(),
                 'anon': tsd.dfAnon.loc[:, 'diff'].std(),
                 },
        'min': {'orig': tsd.dfOrig.loc[:, 'diff'].min(),
                 'anon': tsd.dfAnon.loc[:, 'diff'].min(),
                 },
        'max': {'orig': tsd.dfOrig.loc[:, 'diff'].max(),
                 'anon': tsd.dfAnon.loc[:, 'diff'].max(),
                 },
    }
    tsd.saveStats()

# Test relationship between pickup and dropoff geo-distances
columns = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
tsd = testSynDiffixPy(df, csvFile, columns, output_dir, aidsColumns=aidsColumns, force=False)
if tsd.dfAnon is not None:
    tsd.dfOrig['dist'] = tsd.dfOrig.apply(lambda x: xyDist(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
    tsd.dfAnon['dist'] = tsd.dfAnon.apply(lambda x: xyDist(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
    tsd.moreStats['tripDistance'] = {
        'mean': {'orig': tsd.dfOrig.loc[:, 'dist'].mean(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].mean(),
                 },
        'std': {'orig': tsd.dfOrig.loc[:, 'dist'].std(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].std(),
                 },
        'min': {'orig': tsd.dfOrig.loc[:, 'dist'].min(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].min(),
                 },
        'max': {'orig': tsd.dfOrig.loc[:, 'dist'].max(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].max(),
                 },
    }
    tsd.saveStats()

# Try the above, but this time first convert lat/lon into its S2 cell id, synthesize that,
# and then convert back.
s2s = s2stuff.s2stuff()
df['pickup_cellid'] = df.apply(lambda x: s2s.lat_lon_to_cell_id(x['pickup_latitude'], x['pickup_longitude']), axis=1)
df['dropoff_cellid'] = df.apply(lambda x: s2s.lat_lon_to_cell_id(x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
columns = ['pickup_cellid', 'dropoff_cellid']
tsd = testSynDiffixPy(df, csvFile, columns, output_dir, aidsColumns=aidsColumns, force=False)
if tsd.dfAnon is not None:
    tsd.dfAnon['pickup_latitude'] = tsd.dfAnon.apply(lambda x: s2s.cellId2Lat(x['pickup_cellid']), axis=1)
    tsd.dfAnon['pickup_longitude'] = tsd.dfAnon.apply(lambda x: s2s.cellId2Lon(x['pickup_cellid']), axis=1)
    tsd.dfAnon['dropoff_latitude'] = tsd.dfAnon.apply(lambda x: s2s.cellId2Lat(x['dropoff_cellid']), axis=1)
    tsd.dfAnon['dropoff_longitude'] = tsd.dfAnon.apply(lambda x: s2s.cellId2Lon(x['dropoff_cellid']), axis=1)
    tsd.dfOrig['pickup_latitude'] = tsd.dfOrig.apply(lambda x: s2s.cellId2Lat(x['pickup_cellid']), axis=1)
    tsd.dfOrig['pickup_longitude'] = tsd.dfOrig.apply(lambda x: s2s.cellId2Lon(x['pickup_cellid']), axis=1)
    tsd.dfOrig['dropoff_latitude'] = tsd.dfOrig.apply(lambda x: s2s.cellId2Lat(x['dropoff_cellid']), axis=1)
    tsd.dfOrig['dropoff_longitude'] = tsd.dfOrig.apply(lambda x: s2s.cellId2Lon(x['dropoff_cellid']), axis=1)
    # Now convert the lat/lon into distance, and take stats
    tsd.dfOrig['dist'] = tsd.dfOrig.apply(lambda x: xyDist(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
    tsd.dfAnon['dist'] = tsd.dfAnon.apply(lambda x: xyDist(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
    tsd.moreStats['tripDistance'] = {
        'mean': {'orig': tsd.dfOrig.loc[:, 'dist'].mean(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].mean(),
                 },
        'std': {'orig': tsd.dfOrig.loc[:, 'dist'].std(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].std(),
                 },
        'min': {'orig': tsd.dfOrig.loc[:, 'dist'].min(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].min(),
                 },
        'max': {'orig': tsd.dfOrig.loc[:, 'dist'].max(),
                 'anon': tsd.dfAnon.loc[:, 'dist'].max(),
                 },
    }
    tsd.saveStats()



print("Done!")
