from buckets import Bucket, Bucketizer
from table_generators import TableMaker

# Example usage
table_maker = TableMaker()
table_df = table_maker.simple_uniform(N=1000, C=5, V=5)  # Example dataframe
print(table_df.head())
bucketizer = Bucketizer(df=table_df)
bucketizer.bucketizer(columns=['C1', 'C2'])
bucketizer.bucketizer(columns=['C1'])

# Access bucket information
for col_comb, buckets in bucketizer.buckets_by_combination.items():
    print(f"Buckets for {col_comb}")
    for bucket in buckets:
        print(f"    True Count: {bucket.true_count}")
        print(f"    Noisy Count: {bucket.noisy_count}")
        print(f"    Column Values: {bucket.col_vals}")
        print(f"    Noise Value: {bucket.noise:.2f}")
        print("    -----")