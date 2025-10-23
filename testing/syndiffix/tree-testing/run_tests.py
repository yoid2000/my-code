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

# Generate all combinations
total_combinations = len(skew_values) * len(bumps_values) * len(nrows_values) * len(range_extend_fractions) * len(leafs_modes)
print(f"Running {total_combinations} test combinations...")

combination_count = 0
for skew, bumps, nrows, range_extend_fraction, leafs_mode in itertools.product(skew_values, bumps_values, nrows_values, range_extend_fractions, leafs_modes):
    combination_count += 1
    
    cols = [{'type': 'con', 'skew': skew, 'bumps': bumps}]
    cors = None
    
    print(f"Running combination {combination_count}/{total_combinations}: "
          f"skew={skew}, bumps={bumps}, nrows={nrows}")
    
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

print(f"Completed all {total_combinations} test combinations.")
