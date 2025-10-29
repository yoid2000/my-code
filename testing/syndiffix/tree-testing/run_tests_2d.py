import sys
sys.path.append(r"C:\paul\GitHub\my-code\testing\syndiffix\df-builders")
from leaf_stuff import *
from test_stuff import *
import pprint
import itertools

display_counts = False

pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)

# Define parameter combinations
skew_values = ['strong', 'weak', 'none']
bumps_values = [1, 2, 0, 3]
nrows_values = [1024, 512, 128]
range_extend_fractions = [0.0, 0.15, 0.3, 0.45]
leafs_modes = ["none", "simple", "leaf_only"]
nuniqs = [5, 10, 20]

nrows = 1024
cols = [
    #{'type': 'con', 'skew': 'strong', 'bumps': 2},
    {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': 'strong', 'bumps': 1},
    {'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': 'strong', 'bumps': 1},
    #{'type': 'cat', 'skew': 'none', 'nuniq': 2},
]
cors = [
    [[0, 1], 'weak'],    # ['none', 'weak', 'strong', 'perfect']
]
leaf_mode = "leaf_only"

results, problem_found = run_test(nrows, cors, cols, range_extend_fraction=0.0, leafs_mode=leaf_mode, dump_nodes=True, display_counts=display_counts)

quit()

# Generate all combinations
total_combinations = len(skew_values) * len(bumps_values) * len(nrows_values) * len(range_extend_fractions) * len(leafs_modes)
print(f"Running {total_combinations} test combinations...")


# The following is for categorical data types
print("="*80)
print("Starting tests for categorical data types...")
combination_count = 0
for skew, nuniq, nrows, range_extend_fraction, leafs_mode in itertools.product(skew_values, nuniqs, nrows_values, range_extend_fractions, leafs_modes):
    combination_count += 1
    
    cols = [{'type': 'cat', 'nuniq': nuniq, 'skew': skew}]
    cors = None
    
    print(f"Running combination {combination_count}/{total_combinations}: "
          f"type=cat, skew={skew}, nuniq={nuniq}, nrows={nrows}, range_extend_fraction={range_extend_fraction}, leafs_mode={leafs_mode}")
    
    try:
        results, problem_found = run_test(nrows, cors, cols, range_extend_fraction=range_extend_fraction, leafs_mode=leafs_mode, dump_nodes=False)
        if results is None:
            print(f"  SKIPPED: Results file already exists for combination {combination_count}")
            continue
        if problem_found:
            print(f"  WARNING: Problems found in combination {combination_count}")
        else:
            print(f"  SUCCESS: Combination {combination_count} completed")
    except Exception as e:
        print(f"  ERROR: Exception in combination {combination_count}: {e}")

print(f"Completed all {total_combinations} test combinations for categorical.")

quit()

# The following is for hybrid data types
print("="*80)
print("Starting tests for hybrid data types...")
combination_count = 0
for skew, bumps, nrows, range_extend_fraction, leafs_mode in itertools.product(skew_values, bumps_values, nrows_values, range_extend_fractions, leafs_modes):
    combination_count += 1
    
    cols = [{'type': 'hybrid', 'nuniq': 3, 'point_fraction': 0.2, 'skew': skew, 'bumps': bumps}]
    cors = None
    
    print(f"Running combination {combination_count}/{total_combinations}: "
          f"type=hybrid, skew={skew}, bumps={bumps}, nrows={nrows}, range_extend_fraction={range_extend_fraction}, leafs_mode={leafs_mode}")
    
    try:
        results, problem_found = run_test(nrows, cors, cols, range_extend_fraction=range_extend_fraction, leafs_mode=leafs_mode, dump_nodes=False)
        if results is None:
            print(f"  SKIPPED: Results file already exists for combination {combination_count}")
            continue
        if problem_found:
            print(f"  WARNING: Problems found in combination {combination_count}")
        else:
            print(f"  SUCCESS: Combination {combination_count} completed")
    except Exception as e:
        print(f"  ERROR: Exception in combination {combination_count}: {e}")

print(f"Completed all {total_combinations} test combinations for hybrid.")

quit()

# The following is for continuous data types
print("="*80)
print("Starting tests for continuous data types...")
combination_count = 0
for skew, bumps, nrows, range_extend_fraction, leafs_mode in itertools.product(skew_values, bumps_values, nrows_values, range_extend_fractions, leafs_modes):
    combination_count += 1
    
    cols = [{'type': 'con', 'skew': skew, 'bumps': bumps}]
    cors = None
    
    print(f"Running combination {combination_count}/{total_combinations}: "
          f"type=con, skew={skew}, bumps={bumps}, nrows={nrows}, range_extend_fraction={range_extend_fraction}, leafs_mode={leafs_mode}")
    
    try:
        results, problem_found = run_test(nrows, cors, cols, range_extend_fraction=range_extend_fraction, leafs_mode=leafs_mode, dump_nodes=False)
        if results is None:
            print(f"  SKIPPED: Results file already exists for combination {combination_count}")
            continue
        if problem_found:
            print(f"  WARNING: Problems found in combination {combination_count}")
        else:
            print(f"  SUCCESS: Combination {combination_count} completed")
    except Exception as e:
        print(f"  ERROR: Exception in combination {combination_count}: {e}")

print(f"Completed all {total_combinations} test combinations for continuous.")