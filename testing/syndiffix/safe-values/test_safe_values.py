import sys
import pandas as pd
import numpy as np
import os
from syndiffix import Synthesizer
from syndiffix.stitcher import stitch

# Valid safe_type are 'int', 'str', 'float'
safe_type = 'float'


def do_build():
    """
    Creates a dataframe with 10000 rows and three columns: 'safe', 'cont', and 'cat'.
    
    'safe' column:
    - 250 distinct values total
    - 50 values type1 (1 row each)
    - 50 values type2 (2 rows each) 
    - 50 values type4 (4 rows each)
    - 50 values type8 (8 rows each)
    - 50 values typex (remaining rows)
    
    Data type determined by global safe_type:
    - 'str': '1_001' format (current behavior)
    - 'int': type1: 100-199, type2: 200-299, type4: 400-499, type8: 800-899, typex: 900-999
    - 'float': type1: 1.0-1.99999, type2: 2.0-2.99999, type4: 4.0-4.99999, type8: 8.0-8.99999, typex: 9.0-9.99999
    
    'cont' column: Normal distribution (mean=100, stddev=5)
    'cat' column: 10 distinct string values, uniformly distributed
    """
    global safe_type
    
    # Calculate rows for each group
    rows_1 = 50 * 1  # 50 values, 1 row each = 50 rows
    rows_2 = 50 * 2  # 50 values, 2 rows each = 100 rows
    rows_4 = 50 * 4  # 50 values, 4 rows each = 200 rows
    rows_8 = 50 * 8  # 50 values, 8 rows each = 400 rows
    rows_x = 10000 - (rows_1 + rows_2 + rows_4 + rows_8)  # Remaining rows = 9250
    
    # Generate safe values based on safe_type
    safe_values = []
    
    if safe_type == 'str':
        # String format (current behavior)
        # 1_ values (50 values, 1 row each)
        for i in range(50):
            safe_values.append(f'1_{i:03d}')
        
        # 2_ values (50 values, 2 rows each)
        for i in range(50):
            safe_values.extend([f'2_{i:03d}'] * 2)
        
        # 4_ values (50 values, 4 rows each)
        for i in range(50):
            safe_values.extend([f'4_{i:03d}'] * 4)
        
        # 8_ values (50 values, 8 rows each)
        for i in range(50):
            safe_values.extend([f'8_{i:03d}'] * 8)
        
        # x_ values (50 values, remaining rows distributed)
        rows_per_x = rows_x // 50  # 185 rows per x_ value
        remaining_rows = rows_x % 50  # Any leftover rows
        
        for i in range(50):
            rows_for_this_x = rows_per_x + (1 if i < remaining_rows else 0)
            safe_values.extend([f'x_{i:03d}'] * rows_for_this_x)
    
    elif safe_type == 'int':
        # Integer format
        # type1: 100-199 (50 values, 1 row each)
        for i in range(50):
            safe_values.append(100 + i)
        
        # type2: 200-299 (50 values, 2 rows each)
        for i in range(50):
            safe_values.extend([200 + i] * 2)
        
        # type4: 400-499 (50 values, 4 rows each)
        for i in range(50):
            safe_values.extend([400 + i] * 4)
        
        # type8: 800-899 (50 values, 8 rows each)
        for i in range(50):
            safe_values.extend([800 + i] * 8)
        
        # typex: 900-999 (50 values, remaining rows distributed)
        rows_per_x = rows_x // 50  # 185 rows per x_ value
        remaining_rows = rows_x % 50  # Any leftover rows
        
        for i in range(50):
            rows_for_this_x = rows_per_x + (1 if i < remaining_rows else 0)
            safe_values.extend([900 + i] * rows_for_this_x)
    
    elif safe_type == 'float':
        # Float format
        # type1: 1.0-1.99999 (50 values, 1 row each)
        for i in range(50):
            safe_values.append(1.0 + i * 0.02)  # 1.0, 1.02, 1.04, ..., 1.98
        
        # type2: 2.0-2.99999 (50 values, 2 rows each)
        for i in range(50):
            safe_values.extend([2.0 + i * 0.02] * 2)
        
        # type4: 4.0-4.99999 (50 values, 4 rows each)
        for i in range(50):
            safe_values.extend([4.0 + i * 0.02] * 4)
        
        # type8: 8.0-8.99999 (50 values, 8 rows each)
        for i in range(50):
            safe_values.extend([8.0 + i * 0.02] * 8)
        
        # typex: 9.0-9.99999 (50 values, remaining rows distributed)
        rows_per_x = rows_x // 50  # 185 rows per x_ value
        remaining_rows = rows_x % 50  # Any leftover rows
        
        for i in range(50):
            rows_for_this_x = rows_per_x + (1 if i < remaining_rows else 0)
            safe_values.extend([9.0 + i * 0.02] * rows_for_this_x)
    
    else:
        raise ValueError(f"Invalid safe_type: {safe_type}. Must be 'str', 'int', or 'float'")
    
    # Generate cont values (normal distribution)
    np.random.seed(42)  # For reproducibility
    cont_values = np.random.normal(100, 5, 10000)
    
    # Generate cat values (10 distinct values, uniformly distributed)
    cat_categories = [f'cat_{i}' for i in range(10)]
    cat_values = np.random.choice(cat_categories, 10000)
    
    # Create dataframe
    df = pd.DataFrame({
        'safe': safe_values,
        'cont': cont_values,
        'cat': cat_values
    })
    
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Create directory for safe_type if it doesn't exist
    os.makedirs(safe_type, exist_ok=True)
    
    # Save as parquet in the appropriate directory
    output_path = os.path.join(safe_type, 'original.parquet')
    df.to_parquet(output_path, index=False)
    print(f"Created dataframe with {len(df)} rows and saved as '{output_path}'")
    print(f"Safe type: {safe_type}")
    print(f"Unique 'safe' values: {df['safe'].nunique()}")


def do_synth():
    """
    Synthesize function - creates synthetic dataframes from original.parquet for different column combinations.
    Creates safe and unsafe versions for each combination:
    1. 'safe' only
    2. 'safe' and 'cat' 
    3. 'safe', 'cat', and 'cont' (all columns)
    """
    global safe_type
    
    # Create directory for safe_type if it doesn't exist
    os.makedirs(safe_type, exist_ok=True)
    
    print(f"Reading original.parquet from {safe_type} directory...")
    original_path = os.path.join(safe_type, 'original.parquet')
    df_original = pd.read_parquet(original_path)
    
    # Define column combinations and their file suffixes
    combinations = [
        (['safe'], 'safe_only'),
        (['safe', 'cat'], 'safe_cat'),
        (['safe', 'cat', 'cont'], 'all_cols')
    ]

    # Create the left dataframe for stitching
    print("Creating left dataframe for stitching...")
    df_left = Synthesizer(df_original[['safe']], value_safe_columns=['safe']).sample()
    
    for syn_columns, suffix in combinations:
        print(f"\n--- Processing combination: {syn_columns} ---")
        
        # Create file names with directory path
        safe_file = os.path.join(safe_type, f'synthetic_safe_{suffix}.parquet')
        unsafe_file = os.path.join(safe_type, f'synthetic_unsafe_{suffix}.parquet')
        
        # Check if unsafe synthetic file already exists
        if os.path.exists(unsafe_file):
            print(f"{unsafe_file} already exists, skipping unsafe synthesis...")
        else:
            print(f"Creating unsafe synthetic dataframe for {syn_columns}...")
            df_unsafe = Synthesizer(df_original[syn_columns]).sample()
            df_unsafe.to_parquet(unsafe_file, index=False)
            print(f"Saved {unsafe_file}")
        
        print(f"Creating safe synthetic dataframe for {syn_columns}...")
        df_safe = Synthesizer(df_original[syn_columns], value_safe_columns=['safe']).sample()
        if len(df_safe.columns) > 1:
            print(f"Stitching {df_left.columns} with {df_safe.columns}...")
            df_stitch = stitch(df_left=df_left, df_right=df_safe, shared=False)
        else:
            df_stitch = df_safe
        df_stitch.to_parquet(safe_file, index=False)
        print(f"Saved {safe_file}")
    
    print(f"\nOriginal dataframe: {len(df_original)} rows")
    print("All synthetic dataframes created successfully")


def get_safe_category(value, safe_type):
    """
    Determine which category a safe value belongs to based on safe_type.
    """
    if safe_type == 'str':
        return str(value)[0]  # First character
    elif safe_type == 'int':
        if 100 <= value < 200:
            return '1'
        elif 200 <= value < 300:
            return '2'
        elif 400 <= value < 500:
            return '4'
        elif 800 <= value < 900:
            return '8'
        elif 900 <= value < 1000:
            return 'x'
        else:
            return 'unknown'
    elif safe_type == 'float':
        if 1.0 <= value < 2.0:
            return '1'
        elif 2.0 <= value < 3.0:
            return '2'
        elif 4.0 <= value < 5.0:
            return '4'
        elif 8.0 <= value < 9.0:
            return '8'
        elif 9.0 <= value < 10.0:
            return 'x'
        else:
            return 'unknown'
    else:
        return 'unknown'


def analyze_combination(df_original, df_safe, df_unsafe, combination_name):
    """
    Analyze a specific combination of safe/unsafe synthetic data against original.
    """
    global safe_type
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR: {combination_name.upper()}")
    print(f"{'='*60}")
    
    # 1. Print number of distinct values in 'safe' column for each dataset
    print("\n=== Distinct 'safe' values count ===")
    print(f"Original: {df_original['safe'].nunique()} distinct values")
    print(f"Synthetic Safe: {df_safe['safe'].nunique()} distinct values")
    print(f"Synthetic Unsafe: {df_unsafe['safe'].nunique()} distinct values")
    
    # 2. For each synthetic dataset, find original values missing in synthetic
    original_safe_counts = df_original['safe'].value_counts()
    safe_safe_counts = df_safe['safe'].value_counts()
    unsafe_safe_counts = df_unsafe['safe'].value_counts()
    
    original_values = set(df_original['safe'].unique())
    safe_values = set(df_safe['safe'].unique())
    unsafe_values = set(df_unsafe['safe'].unique())
    
    # Find values in original but not in synthetic_safe
    missing_in_safe = original_values - safe_values
    print(f"\n=== Values in original but missing in synthetic_safe ({len(missing_in_safe)} values) ===")
    if missing_in_safe:
        for value in sorted(missing_in_safe):
            original_count = original_safe_counts[value]
            synthetic_count = safe_safe_counts.get(value, 0)
            print(f"  {value}: original={original_count}, synthetic_safe={synthetic_count}")
    else:
        print("  No missing values")
    
    # Find values in original but not in synthetic_unsafe
    missing_in_unsafe = original_values - unsafe_values
    print(f"\n=== Values in original but missing in synthetic_unsafe ({len(missing_in_unsafe)} values) ===")
    if missing_in_unsafe:
        for value in sorted(missing_in_unsafe):
            original_count = original_safe_counts[value]
            synthetic_count = unsafe_safe_counts.get(value, 0)
            print(f"  {value}: original={original_count}, synthetic_unsafe={synthetic_count}")
    else:
        print("  No missing values")
    
    # Find values in synthetic_safe but not in original
    extra_in_safe = safe_values - original_values
    print(f"\n=== Values in synthetic_safe but missing in original ({len(extra_in_safe)} values) ===")
    if extra_in_safe:
        for value in sorted(extra_in_safe):
            original_count = original_safe_counts.get(value, 0)
            synthetic_count = safe_safe_counts[value]
            print(f"  {value}: original={original_count}, synthetic_safe={synthetic_count}")
    else:
        print("  No extra values")
    
    # Find values in synthetic_unsafe but not in original
    extra_in_unsafe = unsafe_values - original_values
    print(f"\n=== Values in synthetic_unsafe but missing in original ({len(extra_in_unsafe)} values) ===")
    if extra_in_unsafe:
        for value in sorted(extra_in_unsafe):
            original_count = original_safe_counts.get(value, 0)
            synthetic_count = unsafe_safe_counts[value]
            print(f"  {value}: original={original_count}, synthetic_unsafe={synthetic_count}")
    else:
        print("  No extra values")
    
    # Analyze distribution by category
    print(f"\n=== Row counts for {combination_name} ===")
    
    # Get categories and count rows
    original_categories = df_original['safe'].apply(lambda x: get_safe_category(x, safe_type))
    safe_categories = df_safe['safe'].apply(lambda x: get_safe_category(x, safe_type))
    unsafe_categories = df_unsafe['safe'].apply(lambda x: get_safe_category(x, safe_type))
    
    original_cat_counts = original_categories.value_counts().sort_index()
    safe_cat_counts = safe_categories.value_counts().sort_index()
    unsafe_cat_counts = unsafe_categories.value_counts().sort_index()
    
    # Get all unique categories across all datasets
    all_cats = set(original_cat_counts.index) | set(safe_cat_counts.index) | set(unsafe_cat_counts.index)
    
    print(f"Category distribution ({safe_type} type):")
    print(f"{'Cat':<6} {'Original':<10} {'Safe':<15} {'Unsafe':<15}")
    print("-" * 52)
    
    for cat in sorted(all_cats):
        orig_count = original_cat_counts.get(cat, 0)
        safe_count = safe_cat_counts.get(cat, 0)
        unsafe_count = unsafe_cat_counts.get(cat, 0)
        
        safe_diff = safe_count - orig_count
        unsafe_diff = unsafe_count - orig_count
        
        safe_diff_str = f"({safe_diff:+d})" if safe_diff != 0 else ""
        unsafe_diff_str = f"({unsafe_diff:+d})" if unsafe_diff != 0 else ""
        
        print(f"{cat:<6} {orig_count:<10} {safe_count:<6}{safe_diff_str:<9} {unsafe_count:<6}{unsafe_diff_str:<9}")
    
    # Summary totals
    total_safe_diff = len(df_safe) - len(df_original)
    total_unsafe_diff = len(df_unsafe) - len(df_original)
    
    total_safe_diff_str = f"({total_safe_diff:+d})" if total_safe_diff != 0 else ""
    total_unsafe_diff_str = f"({total_unsafe_diff:+d})" if total_unsafe_diff != 0 else ""
    
    print("-" * 52)
    print(f"{'Total':<6} {len(df_original):<10} {len(df_safe):<6}{total_safe_diff_str:<9} {len(df_unsafe):<6}{total_unsafe_diff_str:<9}")


def do_analyze():
    """
    Analyze function - compares 'safe' column values between original and synthetic datasets
    for all column combinations.
    """
    global safe_type
    
    print(f"Reading parquet files from {safe_type} directory...")
    original_path = os.path.join(safe_type, 'original.parquet')
    df_original = pd.read_parquet(original_path)
    
    # Define combinations to analyze
    combinations = [
        ('safe_only', 'Safe Column Only'),
        ('safe_cat', 'Safe and Cat Columns'),
        ('all_cols', 'All Columns (Safe, Cat, Cont)')
    ]
    
    for suffix, name in combinations:
        try:
            safe_path = os.path.join(safe_type, f'synthetic_safe_{suffix}.parquet')
            unsafe_path = os.path.join(safe_type, f'synthetic_unsafe_{suffix}.parquet')
            
            df_safe = pd.read_parquet(safe_path)
            df_unsafe = pd.read_parquet(unsafe_path)
            
            analyze_combination(df_original, df_safe, df_unsafe, name)
            
        except FileNotFoundError as e:
            print(f"\nError: Could not find files for {name} - {e}")
            print("Make sure to run 'synthesize' command first")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_safe_values.py <build|synthesize|analyze>")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "build":
        do_build()
    elif command == "synthesize":
        do_synth()
    elif command == "analyze":
        do_analyze()
    else:
        print("Invalid command. Use 'build', 'synthesize', or 'analyze'")
        sys.exit(1)


if __name__ == "__main__":
    main()
