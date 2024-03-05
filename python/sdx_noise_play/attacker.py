from buckets import Bucket, make_col_vals_key, col_vals_key_to_list, cols_exclude_col_vals
import statistics

def col_subset(combination, col_vals):
    return all(col in combination for col in col_vals)

def col_val_subset(combination, col_vals):
    return all(tuple([col,val]) in combination for col, val in col_vals.items())

class Attacker:
    def __init__(self):
        """
        Initializes an Attacker instance.
        """
        self.buckets = []
        self.buckets_by_combination = {}  # Dictionary to store column combinations

    def add_buckets_by_combination(self, buckets_by_combination):
        self.buckets_by_combination = buckets_by_combination

    def add_bucket(self, bucket):
        """
        Adds a Bucket instance to the list of buckets and organizes by column combinations.
        Args:
            bucket (Bucket): A Bucket object.
        """
        # Create a unique key for the column combination
        key = make_col_vals_key(bucket.col_vals)
        
        # Append the bucket to the associated list (create a new list if not already present)
        self.buckets_by_combination.setdefault(key, []).append(bucket)

    def avg_med_count(self, col_vals):
        ''' For each combination of columns excluding those in col_vals, we want
            to sum up all of bucket noisy counts. This represents one "sample" of
            all rows containing col_vals
        '''
        col_combs = {}
        for combination, bucket_list in self.buckets_by_combination.items():
            if col_val_subset(combination, col_vals):
                col_comb = tuple(cols_exclude_col_vals(combination, col_vals))
                if len(bucket_list) > 1:
                    print(f"Unexpected bucket_list, {combination}, {col_vals}")
                    quit()
                col_combs.setdefault(col_comb, []).append(bucket_list[0].noisy_count)
        total_noisy_counts = []
        for noisy_counts in col_combs.values():
            total_noisy_counts.append(sum(noisy_counts))

        # Calculate overall average noisy_count
        if len(total_noisy_counts) > 0:
            overall_avg_noisy_count = sum(total_noisy_counts) / len(total_noisy_counts)
            median_noisy_count = statistics.median(total_noisy_counts)
            return round(overall_avg_noisy_count), median_noisy_count
        else:
            return 0, 0  # No matching combinations found

if __name__ == "__main__":
    # Example usage
    buckets = []
    buckets.append(Bucket(true_count=100, col_vals={'C1': 1, 'C2': 10}))
    buckets.append(Bucket(true_count=80, col_vals={'C1': 2, 'C2': 20}))
    buckets.append(Bucket(true_count=80, col_vals={'C1': 2, 'C2': 30}))
    buckets.append(Bucket(true_count=80, col_vals={'C1': 2, 'C2': 30, 'C3': 40}))
    buckets.append(Bucket(true_count=120, col_vals={'C1': 2}))
    attacker = Attacker()
    for bucket in buckets:
        bucket.add_noise()
        attacker.add_bucket(bucket)

    if col_val_subset((('C1',1),), {'C1':1}) is False:
        print("BAD: col_val_subset((('C1',1),), {'C1':1}) == False:")
    if col_val_subset((('C1',1),), {'C1':2}) is True:
        print("BAD: col_val_subset((('C1',1),), {'C1':2}) == True:")
    if col_val_subset((('C1',1),('C2',2),), {'C1':2}) is True:
        print("BAD: col_val_subset((('C1',1),('C2',2),), {'C1':2}) == True:")
    if col_val_subset((('C1',1),('C2',2),), {'C1':1}) is False:
        print("BAD: col_val_subset((('C1',1),('C2',2),), {'C1':1}) == False:")
    if col_val_subset((('C1',1),('C2',2),), {'C1':1,'C2':2}) is False:
        print("BAD: col_val_subset((('C1',1),('C2',2),), {'C1':1,'C2':2}) == False:")
    if col_val_subset((('C1',1),('C2',2),), {'C1':1,'C2':3}) is True:
        print("BAD: col_val_subset((('C1',1),('C2',2),), {'C1':1,'C2':2}) == True:")

    # Calculate average count for specific column values
    print(attacker.buckets_by_combination)
    target_col_vals = {'C1': 1, 'C2': 10}
    result_count = attacker.avg_med_count(col_vals=target_col_vals)
    print(f"Average Count for {target_col_vals}: {result_count}")
