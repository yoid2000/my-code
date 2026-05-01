# `gather.py`

`gather.py` builds `dma/raw.parquet` from the AOL source files matching:

- `user-ct-test-collection*.txt`

It loads all matching files into one DataFrame, applies cleaning and de-duplication rules, and writes a sorted parquet output.

# Input Data

The expected source format is tab-separated with columns:

- `AnonID`
- `Query`
- `QueryTime`
- `ItemRank`
- `ClickURL`

Fields may be missing.

# Processing Rules

`gather.py` applies the following rules:

1. Coerce missing fields to nulls.
2. Compress adjacent duplicate runs by key:
   - (`AnonID`, `Query`, `ItemRank`, `ClickURL`)
   - keep only the last record in each adjacent run
   - treat missing values as equal for duplicate comparison
3. Drop records where any of these are missing:
   - `AnonID`
   - `Query`
   - `QueryTime`
4. Type columns as:
   - `AnonID`: int
   - `Query`: text
   - `QueryTime`: datetime
   - `ItemRank`: int
   - `ClickURL`: text
5. Sort output by `QueryTime` ascending.

# `raw.parquet`

Output file:

- `dma/raw.parquet`

This file is the cleaned, typed, de-duplicated, and time-sorted dataset built from all input files.

# Script Output Metrics

When `gather.py` runs, it prints:

- number of adjacent duplicate groups compressed
- number of individual records dropped by that compression
- final number of rows written to `dma/raw.parquet`

The run on the AOL files produced the following:

- Compressed adjacent duplicate groups: 3,360,668
- Dropped records from compression: 6,476,664
- Wrote 29,911,884 rows to raw.parquet