import random
import pandas as pd

def make_col_vals_key(col_vals):
    return tuple(sorted(col_vals.items()))

def col_vals_key_to_list(col_vals_key):
    return list(map(list, zip(*col_vals_key)))

def count_rows_by_col_vals(df, col_vals):
    mask = df.apply(lambda row: all(row[col] == val for col, val in col_vals.items()), axis=1)
    return sum(mask)

def cols_exclude_col_vals(combination, col_vals):
    comb_cols = [comb[0] for comb in combination]
    col_vals_combs = list(col_vals.keys())
    for col in col_vals_combs:
        comb_cols.remove(col)
    return comb_cols


class Bucket:
    def __init__(self, true_count, col_vals):
        """
        Initializes a Bucket instance.
        Args:
            true_count (int): The true count (integer value).
            col_vals (dict): A dictionary where keys are column names and values are volume values.
        """
        self.true_count = true_count
        self.noisy_count = None
        self.noise = None
        self.col_vals = col_vals

    def add_noise(self, std_dev=1.0):
        """
        Adds noise to the true count.
        Args:
            std_dev (float): Standard deviation for generating noise.
        """
        # Generate Gaussian noise with mean 0 and specified standard deviation
        # This emulates two noise layers
        self.noise = random.gauss(0, std_dev)
        self.noise += random.gauss(0, std_dev)
        
        # Calculate noisy count (rounded to nearest integer)
        self.noisy_count = round(self.true_count + self.noise)

    def print(self):
        print(f"    {self.col_vals}, tc={self.true_count}, nc={self.noisy_count}")

class Bucketizer:
    def __init__(self, df, std_dev=1.0):
        """
        Initializes a Bucketizer instance.
        Args:
            df (pd.DataFrame): The dataframe produced by TableMaker.
        """
        self.df = df
        self.buckets_by_combination = {}
        self.std_dev = std_dev

    def bucketizer(self, columns):
        """
        Creates separate Bucket objects for each distinct combination of values for specified columns.
        Args:
            columns (list): List of column names in self.df.
        """
        # Group rows by unique combinations of specified columns
        grouped = self.df.groupby(columns)
        for group_key, group_df in grouped:
            true_count = len(group_df)
            if len(columns) == 1:
                col_vals = {columns[0]: group_key}
            else:
                col_vals = {col: group_key[i] for i, col in enumerate(columns)}
            
            # Create a new Bucket instance
            bucket = Bucket(true_count=true_count, col_vals=col_vals)
            bucket.add_noise(std_dev=self.std_dev)
            col_vals_key = make_col_vals_key(col_vals)
            
            self.buckets_by_combination.setdefault(col_vals_key, []).append(bucket)

    def print(self):
        for col_vals_key, buckets in self.buckets_by_combination.items():
            print(f"{col_vals_key}:")
            for bucket in buckets:
                bucket.print()

if __name__ == "__main__":
    # Example usage
    col_vals = {'C1': 10, 'C2': 20, 'C3': 15}
    true_count_value = 100
    bucket_instance = Bucket(true_count=true_count_value, col_vals=col_vals)
    bucket_instance.add_noise(std_dev=2.5)  # Example standard deviation
    print(f"True Count: {bucket_instance.true_count}")
    print(f"Noisy Count: {bucket_instance.noisy_count}")
    print(f"Noise Value: {bucket_instance.noise:.2f}")

    from table_generators import TableMaker
    N=1000
    C=5
    V=5
    tm = TableMaker()
    df = tm.simple_uniform(N=N, C=C, V=V)  # Example dataframe
    bucketizer = Bucketizer(df=df, std_dev = 0)
    bucketizer.bucketizer(columns=['C1'])
    bucketizer.bucketizer(columns=['C1', 'C2'])
    print(bucketizer.buckets_by_combination)