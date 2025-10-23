#!/usr/bin/env python3
"""
Simple test script to validate basic functionality.
"""

# Test if imports work
try:
    import pandas as pd
    import numpy as np
    from scipy.optimize import linprog
    print("✓ All required packages imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install required packages: pip install pandas numpy scipy")
    exit(1)

# Test basic functionality
from common import make_df, make_bins, add_ran_noise
from solve import solve_noise

def simple_test():
    print("\nRunning simple functionality test...")
    
    # Test parameters
    nu1, nu2, nrows = 3, 3, 20
    
    # Create dataframe
    df = make_df(nu1, nu2, nrows)
    print(f"✓ Created dataframe with shape {df.shape}")
    
    # Make bins
    df_c1, df_c2, df_c1c2 = make_bins(df)
    print(f"✓ Created bins: c1({len(df_c1)}), c2({len(df_c2)}), c1c2({len(df_c1c2)})")
    
    # Add noise (sd=0 for deterministic test)
    df_c1_noisy = add_ran_noise(df_c1, 0.0)
    df_c2_noisy = add_ran_noise(df_c2, 0.0)
    print("✓ Added noise to count data")
    
    # Solve
    try:
        result = solve_noise(df_c1_noisy, df_c2_noisy, df_c1c2, seed=42)
        print(f"✓ Solver completed with {len(result)} combinations")
        
        # Check constraints
        max_noise = max(abs(x) for x in result['noise'])
        print(f"  Maximum absolute noise: {max_noise}")
        
        # Validate sum constraints
        valid = True
        
        # Check c1 constraints
        for _, row in df_c1_noisy.iterrows():
            c1_val = row['values']
            expected = row['count']
            actual = sum(r['count_noise'] for _, r in result.iterrows() if r['values'][0] == c1_val)
            if actual != expected:
                print(f"  ✗ C1 constraint violated: c1={c1_val}, expected={expected}, actual={actual}")
                valid = False
        
        # Check c2 constraints  
        for _, row in df_c2_noisy.iterrows():
            c2_val = row['values']
            expected = row['count']
            actual = sum(r['count_noise'] for _, r in result.iterrows() if r['values'][1] == c2_val)
            if actual != expected:
                print(f"  ✗ C2 constraint violated: c2={c2_val}, expected={expected}, actual={actual}")
                valid = False
        
        if valid:
            print("✓ All constraints satisfied")
        else:
            print("✗ Some constraints violated")
            
    except Exception as e:
        print(f"✗ Solver failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
    print("\nSimple test completed!")