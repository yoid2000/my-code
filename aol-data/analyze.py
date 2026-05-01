import os
import pandas as pd

records = []

# Loop through all .txt files in the directory
# but only if data.parquet does not exist
if not os.path.exists('data.parquet'):
    for fname in os.listdir('.'):
        if fname.endswith('.txt') and not fname.startswith('arnold'):
            with open(fname, encoding='utf-8') as f:
                print(f"Processing file: {fname}")
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        continue
                    user_id = parts[0]
                    if not user_id.isdigit():
                        continue
                    query_string = parts[1]
                    query_datetime_raw = parts[2]
                    records.append({
                        'user_id': int(user_id),
                        'query_string': query_string,
                        'query_datetime': query_datetime_raw
                    })

    # Create DataFrame
    df = pd.DataFrame(records)
    df['query_datetime'] = pd.to_datetime(df['query_datetime'], errors='coerce')
    # save to parquet for faster loading next time
    df.to_parquet('data.parquet', index=False)
else:
    print("Loading data from data.parquet")
    df = pd.read_parquet('data.parquet')
    # Backfill for older parquet files that were created before query_datetime was captured
    if 'query_datetime' not in df.columns:
        df['query_datetime'] = pd.NaT
    else:
        df['query_datetime'] = pd.to_datetime(df['query_datetime'], errors='coerce')

print(df.head())
print("Total records:", len(df))

# total number of distinct user_ids
distinct_user_ids = df['user_id'].nunique()
print("Total distinct user_ids:", distinct_user_ids)

# Lowercase query_string for case-insensitive search
df['query_string_lower'] = df['query_string'].str.lower()

# Count rows where both "arnold" and "lilburn" are in query_string
mask_both = df['query_string_lower'].str.contains('arnold') & df['query_string_lower'].str.contains('lilburn')
count_both = mask_both.sum()

# Of those, count where user_id == 4417749
count_both_user = df[mask_both & (df['user_id'] == 4417749)].shape[0]

# Rows where "arnold" in query_string, but user_id != 4417749
mask_arnold = df['query_string_lower'].str.contains('arnold') & (df['user_id'] != 4417749)
count_arnold_not_user = mask_arnold.sum()

# Rows where "lilburn" in query_string, but user_id != 4417749
mask_lilburn = df['query_string_lower'].str.contains('lilburn') & (df['user_id'] != 4417749)
count_lilburn_not_user = mask_lilburn.sum()

# Rows where "thelma" in query_string, and user_id == 4417749
mask_thelma = df['query_string_lower'].str.contains('thelma') & (df['user_id'] == 4417749)
count_thelma_user = mask_thelma.sum()

print("Rows with both 'arnold' and 'lilburn':", count_both)
print("Rows with both 'arnold' and 'lilburn' and user_id==4417749:", count_both_user)
print("Rows with 'arnold' and user_id!=4417749:", count_arnold_not_user)
print("Rows with 'lilburn' and user_id!=4417749:", count_lilburn_not_user)
print("Rows with 'thelma' and user_id==4417749:", count_thelma_user)

# Build the repeats dictionary: sorted term set -> count of distinct user_ids
# Sort words so that "foo bar" and "bar foo" are treated as the same term set
multi_word = df[df['query_string_lower'].str.split().str.len() > 1].copy()
multi_word['term_set'] = multi_word['query_string_lower'].str.split().apply(lambda w: ' '.join(sorted(w)))

repeats = multi_word.groupby('term_set')['user_id'].nunique()

# Count how many queries have more than one distinct user_id
multi_user_count = (repeats > 1).sum()
print("Number of query_string_lower with >1 word and >1 user_id:", multi_user_count)

# For queries with >1 user_id, group by term count and compute stats
multi_user = repeats[repeats > 1].reset_index()
multi_user.columns = ['term_set', 'user_id_count']
multi_user['term_count'] = multi_user['term_set'].str.split().str.len()

stats = (
    multi_user.groupby('term_count')['user_id_count']
    .agg(
        num_queries='count',
        avg='mean',
        median='median',
        stdev='std',
        p90=lambda x: x.quantile(0.90),
        max='max'
    )
    .reset_index()
)

print("\nFor each term count with >1 user_id (stats over user_id counts per query):")
print(f"{'term_count':>12} {'num_queries':>12} {'avg':>8} {'median':>8} {'stdev':>8} {'p90':>8} {'max':>8}")
for _, row in stats.iterrows():
    print(f"{int(row['term_count']):>12} {int(row['num_queries']):>12} {row['avg']:>8.2f} {row['median']:>8.2f} {row['stdev']:>8.2f} {row['p90']:>8.2f} {int(row['max']):>8}")

print("\nExamples of term sets for each term count:")
for _, row in stats.iterrows():
    print(f"{int(row['term_count']):>12} {int(row['num_queries']):>12} {row['avg']:>8.2f} {row['median']:>8.2f} {row['stdev']:>8.2f} {row['p90']:>8.2f} {int(row['max']):>8}")
    examples = multi_user[multi_user['term_count'] == row['term_count']]['term_set'].head(3).tolist()
    for ex in examples:
        print(f"  e.g. {ex}")