import numpy as np
import pandas as pd
from itertools import combinations, product
from collections import defaultdict
import random
import math
from ortools.sat.python import cp_model

tolerance_factor = 2.0

class SolutionCounter(cp_model.CpSolverSolutionCallback):
    """Callback to count the number of solutions."""
    
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.solution_count = 0
        self.max_solutions = 10  # Limit to avoid infinite enumeration
    
    def on_solution_callback(self):
        self.solution_count += 1
        if self.solution_count >= self.max_solutions:
            print(f"Reached maximum solution limit of {self.max_solutions}")
            self.StopSearch()
    
    def solution_count_value(self):
        return self.solution_count

def check_bucket_consistency(noisy_buckets, Cstd, Cmin):
    """
    Check for inconsistencies between parent and child bucket constraints.
    
    A parent bucket contains Y columns, child buckets contain Y+1 columns
    where Y columns match the parent exactly.
    """
    print("Checking bucket consistency...")
    inconsistencies = []
    
    # Group buckets by number of columns
    buckets_by_size = {}
    for bucket_key, noisy_count in noisy_buckets.items():
        col_combo, values = bucket_key
        num_cols = len(col_combo)
        if num_cols not in buckets_by_size:
            buckets_by_size[num_cols] = []
        buckets_by_size[num_cols].append((bucket_key, noisy_count))
    
    # Check each parent-child relationship
    for parent_size in sorted(buckets_by_size.keys()):
        child_size = parent_size + 1
        if child_size not in buckets_by_size:
            continue
            
        parents = buckets_by_size[parent_size]
        children = buckets_by_size[child_size]
        
        for parent_key, parent_noisy_count in parents:
            parent_cols, parent_values = parent_key
            
            # Find all children of this parent
            child_buckets = []
            for child_key, child_noisy_count in children:
                child_cols, child_values = child_key
                
                # Check if this child is a child of the parent
                # Parent columns must be a subset of child columns with same values
                is_child = True
                for i, parent_col in enumerate(parent_cols):
                    parent_val = parent_values[i]
                    # Find this column in the child
                    try:
                        child_idx = child_cols.index(parent_col)
                        child_val = child_values[child_idx]
                        if parent_val != child_val:
                            is_child = False
                            break
                    except ValueError:
                        is_child = False
                        break
                
                if is_child:
                    child_buckets.append((child_key, child_noisy_count))
            
            if not child_buckets:
                continue
                
            # Calculate constraint ranges
            noise_tolerance = tolerance_factor * Cstd
            
            # Parent range
            parent_min = max(Cmin, int(round(parent_noisy_count - noise_tolerance)))
            parent_max = int(round(parent_noisy_count + noise_tolerance))

            # Children ranges
            child_ranges = []
            child_min_sum = 0
            child_max_sum = 0
            
            for child_key, child_noisy_count in child_buckets:
                child_min = max(Cmin, int(round(child_noisy_count - noise_tolerance)))
                child_max = int(round(child_noisy_count + noise_tolerance))
                child_ranges.append((child_key, child_min, child_max))
                child_min_sum += child_min
                child_max_sum += child_max
            
            # Check for inconsistency
            # The sum of children should be consistent with the parent
            inconsistent = False
            reason = ""
            
            if child_min_sum > parent_max:
                inconsistent = True
                reason = f"Children minimum sum ({child_min_sum}) > parent maximum ({parent_max})"
            elif child_max_sum < parent_min:
                inconsistent = True
                reason = f"Children maximum sum ({child_max_sum}) < parent minimum ({parent_min})"
            
            if inconsistent:
                inconsistencies.append({
                    'parent': (parent_key, parent_noisy_count, parent_min, parent_max),
                    'children': child_ranges,
                    'reason': reason,
                    'child_sum_range': (child_min_sum, child_max_sum)
                })
    
    return inconsistencies

def test_reconstruction(N=1000, C=5, U=10, Smean=4.0, Smin=2, Sstd=1.0, Cmean=0, Cstd=1.0, Cmin=0, count_solutions=False):
    """
    Test routine for data reconstruction using constraint solving.
    
    Parameters:
    N: Number of rows in the test dataset
    C: Number of columns in the dataset  
    U: Number of distinct values per column
    Smean: Mean suppression threshold
    Smin: Minimum suppression threshold
    Sstd: Standard deviation of the suppression threshold
    Cmean: Mean of the count noise
    Cstd: Standard deviation of the count noise
    Cmin: Minimum noisy count
    count_solutions: If True, attempt to count multiple solutions
    """
    print(f"Starting reconstruction test with parameters: N={N}, C={C}, U={U}, Smean={Smean}, Smin={Smin}, Sstd={Sstd}, Cmean={Cmean}, Cstd={Cstd}, Cmin={Cmin}")
    # Step 1: Generate synthetic dataset
    print(f"Generating dataset with {N} rows, {C} columns, {U} distinct values per column")
    data = np.random.randint(0, U, size=(N, C))
    original_df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(C)])
    print(f"Original data shape: {original_df.shape}")
    
    # Validate that all rows are unique
    unique_rows = original_df.drop_duplicates()
    if len(unique_rows) != len(original_df):
        duplicate_count = len(original_df) - len(unique_rows)
        raise ValueError(f"Original dataset contains {duplicate_count} duplicate rows. "
                        f"All rows must be unique for reconstruction testing. "
                        f"Consider increasing U (distinct values per column) or C (number of columns).")
    print(f"Verified: All {N} rows are unique")
    
    # Create a second random dataset with same parameters for comparison
    print(f"Generating comparison dataset with same parameters...")
    random_data = np.random.randint(0, U, size=(N, C))
    random_df = pd.DataFrame(random_data, columns=[f'col_{i}' for i in range(C)])
    
    # Count matching rows between original and random datasets
    original_rows_set = set(tuple(row) for row in original_df.values)
    random_rows_set = set(tuple(row) for row in random_df.values)
    random_matched_rows = original_rows_set.intersection(random_rows_set)
    random_accuracy = len(random_matched_rows) / N
    
    print(f"Random baseline: {len(random_matched_rows)} matching rows ({random_accuracy:.2%} accuracy)")
    
    # Step 2: Create buckets for all column combinations
    print("Creating buckets for all column combinations...")
    buckets = {}
    
    for i in range(1, C + 1):
        for col_combo in combinations(range(C), i):
            # Get column names for this combination
            col_names = [f'col_{j}' for j in col_combo]
            
            # Count occurrences of each unique combination of values
            value_counts = original_df.groupby(col_names).size().reset_index(name='count')
            
            # Store buckets
            for _, row in value_counts.iterrows():
                bucket_key = (col_combo, tuple(row[col_names]))
                buckets[bucket_key] = row['count']
    
    print(f"Total buckets created: {len(buckets)}")
    
    # Step 3: Apply suppression thresholds
    print("Applying suppression thresholds...")
    suppressed_buckets = {}
    
    for bucket_key, count in buckets.items():
        # Generate random suppression threshold
        T = max(Smin, np.random.normal(Smean, Sstd))
        
        # Keep bucket only if count > T
        if count > T:
            suppressed_buckets[bucket_key] = count
    
    print(f"Buckets after suppression: {len(suppressed_buckets)}")
    #print(suppressed_buckets)
    
    # Step 4: Add noise to remaining buckets
    print("Adding noise to bucket counts...")
    noisy_buckets = {}
    
    for bucket_key, count in suppressed_buckets.items():
        # Add noise
        noise = np.random.normal(Cmean, Cstd)
        noisy_count = max(Cmin, count + noise)
        noisy_buckets[bucket_key] = int(round(noisy_count))
    
    print(f"Final buckets with noise: {len(noisy_buckets)}")
    
    if False:
        # Check for bucket inconsistencies
        inconsistencies = check_bucket_consistency(noisy_buckets, Cstd, Cmin)
        
        if inconsistencies:
            print(f"\nFound {len(inconsistencies)} bucket inconsistencies:")
            for i, inc in enumerate(inconsistencies):
                print(f"\nInconsistency {i+1}:")
                parent_key, parent_noisy, parent_min, parent_max = inc['parent']
                print(f"  Parent {parent_key}: noisy_count={parent_noisy}, range=[{parent_min}, {parent_max}]")
                print(f"  Children:")
                for child_key, child_min, child_max in inc['children']:
                    print(f"    {child_key}: range=[{child_min}, {child_max}]")
                print(f"  Children sum range: [{inc['child_sum_range'][0]}, {inc['child_sum_range'][1]}]")
                print(f"  Problem: {inc['reason']}")
        else:
            print("\nNo bucket inconsistencies found.")
    
    # Step 5: Reconstruct data using constraint solver with iterative tolerance
    print("Attempting reconstruction using constraint solver...")
    
    current_tolerance_factor = 1.0
    max_tolerance_factor = 10.0  # Reasonable upper limit
    solution_found = False
    
    while current_tolerance_factor <= max_tolerance_factor and not solution_found:
        print(f"\nTrying reconstruction with tolerance_factor = {current_tolerance_factor}")
        
        # Create the constraint programming model
        model = cp_model.CpModel()
        
        # Create decision variables for each cell in the reconstructed dataset
        x = {}
        for i in range(N):
            x[i] = {}
            for j in range(C):
                x[i][j] = model.NewIntVar(0, U-1, f'x_{i}_{j}')
        
        # Add constraints based on the noisy bucket counts
        bucket_stats = {}  # Track statistics by bucket size
        
        for bucket_key, noisy_count in noisy_buckets.items():
            col_combo, values = bucket_key
            bucket_size = len(col_combo)
            
            # Calculate the expected range using current tolerance factor
            noise_tolerance = current_tolerance_factor * Cstd
            expected_min = max(Cmin, math.floor(noisy_count - noise_tolerance))
            expected_max = math.ceil(noisy_count + noise_tolerance)
            range_size = expected_max - expected_min
            
            # Collect statistics
            if bucket_size not in bucket_stats:
                bucket_stats[bucket_size] = {
                    'count': 0,
                    'range_sizes': [],
                    'min_boundaries': [],
                    'max_boundaries': []
                }
            
            bucket_stats[bucket_size]['count'] += 1
            bucket_stats[bucket_size]['range_sizes'].append(range_size)
            bucket_stats[bucket_size]['min_boundaries'].append(expected_min)
            bucket_stats[bucket_size]['max_boundaries'].append(expected_max)
            
            # Count how many rows in our reconstruction match this bucket's pattern
            matching_bools = []
            for i in range(N):
                row_matches = model.NewBoolVar(f'match_{i}_{bucket_key}_{current_tolerance_factor}')
                
                conditions = []
                for idx, col_idx in enumerate(col_combo):
                    required_value = values[idx]
                    col_match = model.NewBoolVar(f'col_{i}_{col_idx}_{required_value}_{bucket_key}_{current_tolerance_factor}')
                    model.Add(x[i][col_idx] == required_value).OnlyEnforceIf(col_match)
                    model.Add(x[i][col_idx] != required_value).OnlyEnforceIf(col_match.Not())
                    conditions.append(col_match)
                
                if len(conditions) == 1:
                    model.Add(row_matches == conditions[0])
                else:
                    model.AddBoolAnd(conditions).OnlyEnforceIf(row_matches)
                    model.AddBoolOr([c.Not() for c in conditions]).OnlyEnforceIf(row_matches.Not())
                
                matching_bools.append(row_matches)
            
            actual_count = sum(matching_bools)
            model.Add(actual_count >= expected_min)
            model.Add(actual_count <= expected_max)
        
        # Print constraint statistics by bucket size
        print("  Constraint statistics by bucket size:")
        for bucket_size in sorted(bucket_stats.keys()):
            stats = bucket_stats[bucket_size]
            range_sizes = stats['range_sizes']
            min_boundaries = stats['min_boundaries']
            max_boundaries = stats['max_boundaries']
            
            avg_range = np.mean(range_sizes)
            std_range = np.std(range_sizes)
            min_range = min(range_sizes)
            max_range = max(range_sizes)
            min_lower = min(min_boundaries)
            max_upper = max(max_boundaries)
            
            print(f"    {bucket_size}-col buckets: {stats['count']} buckets, "
                    f"range size avg={avg_range:.1f} std={std_range:.1f} "
                    f"min={min_range} max={max_range}, "
                    f"bounds=[{min_lower}, {max_upper}]")

        # Calculate how many buckets each row belongs to
        row_bucket_counts = [0] * N
        for bucket_key, noisy_count in noisy_buckets.items():
            col_combo, values = bucket_key
            
            # Check each row to see if it matches this bucket
            for i in range(N):
                matches_bucket = True
                for idx, col_idx in enumerate(col_combo):
                    required_value = values[idx]
                    # Get the actual value from original data for comparison
                    actual_value = original_df.iloc[i, col_idx]
                    if actual_value != required_value:
                        matches_bucket = False
                        break
                
                if matches_bucket:
                    row_bucket_counts[i] += 1
        
        # Print statistics about bucket membership per row
        avg_buckets = np.mean(row_bucket_counts)
        std_buckets = np.std(row_bucket_counts)
        min_buckets = min(row_bucket_counts)
        max_buckets = max(row_bucket_counts)
        
        print(f"  Row bucket membership: avg={avg_buckets:.1f} std={std_buckets:.1f} "
              f"min={min_buckets} max={max_buckets}")

        # Configure and run the solver
        solver = cp_model.CpSolver()
        
        if count_solutions:
            solver.parameters.max_time_in_seconds = 300.0
            solver.parameters.enumerate_all_solutions = True
            
            solution_counter = SolutionCounter()
            status = solver.SearchForAllSolutions(model, solution_counter)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                num_solutions = solution_counter.solution_count_value()
                print(f"Number of solutions found: {num_solutions}")
                
                if num_solutions >= solution_counter.max_solutions:
                    print("WARNING: Solution count may be incomplete due to limit")
                
                total_variables = N * C
                total_constraints = len(noisy_buckets)
                degrees_of_freedom = total_variables - total_constraints
                print(f"Degrees of freedom estimate: {total_variables} variables - {total_constraints} constraints = {degrees_of_freedom}")
                
                theoretical_max = U ** (N * C)
                print(f"Theoretical maximum solutions: {U}^{N*C} = {theoretical_max}")
                
                if num_solutions > 1:
                    print(f"Multiple solutions exist ({num_solutions}), which explains the low reconstruction accuracy")
                
                # Get one solution for detailed analysis
                solver.parameters.enumerate_all_solutions = False
                solver.parameters.max_time_in_seconds = 6000000.0
                status = solver.Solve(model)
        else:
            solver.parameters.max_time_in_seconds = int(120.0 * current_tolerance_factor)
            print(f"Solving with timeout of {solver.parameters.max_time_in_seconds} seconds...")
            status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found with tolerance_factor = {current_tolerance_factor}!")
            solution_found = True
            
            # Extract reconstructed data
            reconstructed_data = np.zeros((N, C), dtype=int)
            for i in range(N):
                for j in range(C):
                    reconstructed_data[i][j] = solver.Value(x[i][j])
            
            reconstructed_df = pd.DataFrame(reconstructed_data, columns=[f'col_{i}' for i in range(C)])
            
            # Compare with original (order-independent)
            original_rows = set(tuple(row) for row in original_df.values)
            reconstructed_rows = set(tuple(row) for row in reconstructed_df.values)
            
            matched_rows = original_rows.intersection(reconstructed_rows)
            accuracy = len(matched_rows) / N
            
            print(f"Reconstruction accuracy (order-independent): {accuracy:.2%}")
            print(f"Original unique rows: {len(original_rows)}")
            print(f"Reconstructed unique rows: {len(reconstructed_rows)}")
            print(f"Matched rows: {len(matched_rows)}")
            print(f"Random baseline accuracy: {random_accuracy:.2%} ({len(random_matched_rows)} matches)")
            
            # Find rows that match on exactly C-1 columns
            near_match_count = 0
            for _, recon_row in reconstructed_df.iterrows():
                for _, orig_row in original_df.iterrows():
                    # Count matching columns
                    matching_cols = sum(recon_row[col] == orig_row[col] for col in original_df.columns)
                    if matching_cols == C - 1:
                        near_match_count += 1
                        break  # Found a near match for this reconstructed row, move to next
            
            print(f"Rows matching on exactly {C-1} columns: {near_match_count}")
            precision = len(matched_rows) / (len(matched_rows) + near_match_count) if (len(matched_rows) + near_match_count) > 0 else 0
            recall = (len(matched_rows) + near_match_count) / N if N > 0 else 0
            print(f"Precision of inferences: {precision:.2%}  (against baseline of {1/U:.2%}")
            print(f"Recall of inferences: {recall:.2%}")

            return original_df, reconstructed_df, accuracy
        else:
            if status == cp_model.INFEASIBLE:
                print(f"INFEASIBLE with tolerance_factor = {current_tolerance_factor}")
            elif status == cp_model.MODEL_INVALID:
                print(f"MODEL_INVALID with tolerance_factor = {current_tolerance_factor}")
            elif status == cp_model.UNKNOWN:
                print(f"UNKNOWN/TIMEOUT with tolerance_factor = {current_tolerance_factor}")
            else:
                print(f"Status {status} with tolerance_factor = {current_tolerance_factor}")
            
            current_tolerance_factor += 1.0
    
    # If we get here, no solution was found even with maximum tolerance
    print(f"\nNo solution found even with maximum tolerance_factor = {max_tolerance_factor}")
    print("The constraints may be fundamentally inconsistent.")
    
    # Print final diagnostic information
    print(f"\nFinal diagnostic information:")
    print(f"  Total variables: {N * C}")
    print(f"  Total constraints: {len(noisy_buckets)}")
    print(f"  Final tolerance used: {max_tolerance_factor} * {Cstd} = {max_tolerance_factor * Cstd}")
    
    return original_df, None, 0.0

if __name__ == "__main__":
    test_reconstruction(N=400, C=5, U=10, Smean=1, Smin=1, Sstd=0, Cmean=0, Cstd=0.5, Cmin=1, count_solutions=False)
    #test_reconstruction(N=1000, C=5, U=10, Smean=4, Smin=2, Sstd=1, Cmean=0, Cstd=1, Cmin=0)