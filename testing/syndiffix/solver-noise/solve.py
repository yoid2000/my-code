import pandas as pd
import numpy as np
from typing import Tuple, Any
from scipy.optimize import linprog
import random


def solve_noise(df_c1: pd.DataFrame, df_c2: pd.DataFrame, df_c1c2: pd.DataFrame, seed: int, constraint_tolerance: float = 1.0, do_print: bool = False) -> pd.DataFrame:
    """
    Solves for noise values that minimize maximum absolute noise while satisfying sum constraints.
    
    Args:
        df_c1: DataFrame with unique c1 values and their counts
        df_c2: DataFrame with unique c2 values and their counts  
        df_c1c2: DataFrame with unique (c1, c2) combinations and their counts
        seed: Random seed for the solver
        constraint_tolerance: Maximum allowed deviation from exact marginal sums (default: 1.0)
        do_print: Whether to print debugging information (default: False)
        
    Returns:
        DataFrame with columns: values, count, count_noise, noise
        where count_noise = count + noise and max absolute noise is minimized
    """
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Create result dataframe starting with df_c1c2
    result_df = df_c1c2.copy()
    
    # Extract c1 and c2 from values tuples
    c1_vals = [val[0] for val in result_df['values']]
    c2_vals = [val[1] for val in result_df['values']]
    
    # Number of variables (one noise variable per combination)
    n_vars = len(result_df)
    
    # We'll solve a linear programming problem to minimize the maximum absolute noise
    # Variables: [noise_1, noise_2, ..., noise_n, max_abs_noise]
    # We add one extra variable for the maximum absolute noise
    n_total_vars = n_vars + 1
    
    # Objective: minimize max_abs_noise (last variable) with small random perturbations for variation
    c = np.zeros(n_total_vars)
    c[-1] = 1  # Minimize the last variable (max_abs_noise)
    
    # Add small random perturbations to the noise variable coefficients to create solution variation
    perturbation_scale = 1e-9  # Reduced from 1e-6 to avoid numerical issues
    for i in range(n_vars):
        c[i] = np.random.normal(0, perturbation_scale)
    
    # Constraints:
    # 1. Sum constraints for c1 values (converted to inequalities)
    # 2. Sum constraints for c2 values (converted to inequalities)
    # 3. Bounds for max_abs_noise: noise_i <= max_abs_noise and -noise_i <= max_abs_noise
    
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    
    # Constraint 1: For each unique c1 value, sum of (count + noise) should be close to target count
    # Convert to inequality: |sum - target| <= constraint_tolerance
    # This becomes: sum - target <= constraint_tolerance AND target - sum <= constraint_tolerance
    unique_c1 = df_c1['values'].tolist()
    for c1_val in unique_c1:
        target_count = df_c1[df_c1['values'] == c1_val]['count'].iloc[0]
        current_sum = 0
        
        # Create constraint vector for this c1 value
        constraint = np.zeros(n_total_vars)
        for i, (c1, c2) in enumerate(zip(c1_vals, c2_vals)):
            if c1 == c1_val:
                constraint[i] = 1  # coefficient for noise_i
                current_sum += result_df.iloc[i]['count']
        
        # Add two inequality constraints: sum - target <= tolerance and target - sum <= tolerance
        # First constraint: sum of (count + noise) - target <= constraint_tolerance
        # Which becomes: sum of noise <= (target - current_sum) + constraint_tolerance
        A_ub.append(constraint.copy())
        b_ub.append(target_count - current_sum + constraint_tolerance)
        
        # Second constraint: target - sum of (count + noise) <= constraint_tolerance  
        # Which becomes: -sum of noise <= (current_sum - target) + constraint_tolerance
        A_ub.append(-constraint.copy())
        b_ub.append(current_sum - target_count + constraint_tolerance)
    
    # Constraint 2: For each unique c2 value, sum of (count + noise) should be close to target count
    unique_c2 = df_c2['values'].tolist()
    for c2_val in unique_c2:
        target_count = df_c2[df_c2['values'] == c2_val]['count'].iloc[0]
        current_sum = 0
        
        # Create constraint vector for this c2 value
        constraint = np.zeros(n_total_vars)
        for i, (c1, c2) in enumerate(zip(c1_vals, c2_vals)):
            if c2 == c2_val:
                constraint[i] = 1  # coefficient for noise_i
                current_sum += result_df.iloc[i]['count']
        
        # Add two inequality constraints: sum - target <= tolerance and target - sum <= tolerance
        # First constraint: sum of (count + noise) - target <= constraint_tolerance
        A_ub.append(constraint.copy())
        b_ub.append(target_count - current_sum + constraint_tolerance)
        
        # Second constraint: target - sum of (count + noise) <= constraint_tolerance
        A_ub.append(-constraint.copy())
        b_ub.append(current_sum - target_count + constraint_tolerance)
    
    # Inequality constraints for max_abs_noise bounds
    
    # For each noise variable: noise_i <= max_abs_noise and -noise_i <= max_abs_noise
    for i in range(n_vars):
        # noise_i - max_abs_noise <= 0
        constraint1 = np.zeros(n_total_vars)
        constraint1[i] = 1
        constraint1[-1] = -1
        A_ub.append(constraint1)
        b_ub.append(0)
        
        # -noise_i - max_abs_noise <= 0
        constraint2 = np.zeros(n_total_vars)
        constraint2[i] = -1
        constraint2[-1] = -1
        A_ub.append(constraint2)
        b_ub.append(0)
    
    # Convert to numpy arrays
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    
    # Print all equations for debugging
    if do_print:
        print(f"\n--- Linear Programming Problem Setup (seed={seed}) ---")
        print(f"Problem size: {len(df_c1c2)} combinations, {len(unique_c1)} c1 values, {len(unique_c2)} c2 values")
        print(f"Variables: {n_vars} noise variables + 1 max_abs_noise variable = {n_total_vars} total")
        print(f"Objective: minimize max_abs_noise (variable {n_vars})")
        print(f"Constraint tolerance: {constraint_tolerance}")
        
        # Print combination mappings
        print(f"\nCombination mappings:")
        for i, (c1, c2) in enumerate(zip(c1_vals, c2_vals)):
            original_count = result_df.iloc[i]['count']
            print(f"  noise_{i}: combination ({c1},{c2}) with original count {original_count}")
        
        # Print marginal sum constraints (now inequalities)
        print(f"\nMarginal sum constraints (inequalities with tolerance {constraint_tolerance}):")
        
        # Print c1 constraints
        for c1_val in unique_c1:
            target_count = df_c1[df_c1['values'] == c1_val]['count'].iloc[0]
            current_sum = sum(result_df.iloc[i]['count'] for i, (c1, c2) in enumerate(zip(c1_vals, c2_vals)) if c1 == c1_val)
            
            constraint_terms = []
            for i, (c1, c2) in enumerate(zip(c1_vals, c2_vals)):
                if c1 == c1_val:
                    constraint_terms.append(f"noise_{i}")
            
            constraint_str = " + ".join(constraint_terms) if constraint_terms else "0"
            ideal_rhs = target_count - current_sum
            print(f"  C1_{c1_val}: |{constraint_str} - ({ideal_rhs})| <= {constraint_tolerance}")
            print(f"    -> {constraint_str} <= {ideal_rhs + constraint_tolerance}")
            print(f"    -> {constraint_str} >= {ideal_rhs - constraint_tolerance}")
        
        # Print c2 constraints
        for c2_val in unique_c2:
            target_count = df_c2[df_c2['values'] == c2_val]['count'].iloc[0]
            current_sum = sum(result_df.iloc[i]['count'] for i, (c1, c2) in enumerate(zip(c1_vals, c2_vals)) if c2 == c2_val)
            
            constraint_terms = []
            for i, (c1, c2) in enumerate(zip(c1_vals, c2_vals)):
                if c2 == c2_val:
                    constraint_terms.append(f"noise_{i}")
            
            constraint_str = " + ".join(constraint_terms) if constraint_terms else "0"
            ideal_rhs = target_count - current_sum
            print(f"  C2_{c2_val}: |{constraint_str} - ({ideal_rhs})| <= {constraint_tolerance}")
            print(f"    -> {constraint_str} <= {ideal_rhs + constraint_tolerance}")
            print(f"    -> {constraint_str} >= {ideal_rhs - constraint_tolerance}")
        
        # Print inequality constraints for max_abs_noise
        print(f"\nAbsolute value constraints ({2 * n_vars} constraints):")
        for i in range(n_vars):
            print(f"  |noise_{i}| <= max_abs_noise:")
            print(f"    noise_{i} - max_abs_noise <= 0")
            print(f"    -noise_{i} - max_abs_noise <= 0")
        
        # Print bounds
        print(f"\nVariable bounds:")
        for i in range(n_vars):
            print(f"  noise_{i}: unbounded")
        print(f"  max_abs_noise: >= 0")
        print("--- End Linear Programming Setup ---\n")
    
    # Bounds: noise variables can be any real number, max_abs_noise >= 0
    bounds = [(None, None)] * n_vars + [(0, None)]
    
    # Solve the linear programming problem
    # Try HiGHS first, fallback to revised simplex if it fails
    solver_options = {
        'presolve': True,
        'disp': False,
        'maxiter': 10000
    }
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=solver_options)
    
    # If HiGHS fails with numerical issues, try revised simplex method
    if not result.success and ("Status 4" in str(result.message) or "solve error" in str(result.message).lower()):
        if do_print:
            print(f"HiGHS failed ({result.message}), trying revised simplex method...")
        
        # Try with revised simplex and different options
        simplex_options = {
            'maxiter': 10000,
            'disp': False,
            'rr': True,  # Use row reduction
            'bland': True  # Use Bland's pivoting rule for numerical stability
        }
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex', options=simplex_options)
        
        # If that also fails, try interior-point
        if not result.success:
            if do_print:
                print(f"Revised simplex also failed ({result.message}), trying interior-point method...")
            interior_options = {
                'maxiter': 10000,
                'disp': False
            }
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point', options=interior_options)
    
    if not result.success:
        if "infeasible" in result.message.lower():
            raise ValueError("INFEASIBLE")
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    # Extract noise values (excluding the max_abs_noise variable)
    noise_values = result.x[:-1]
    
    # Round noise to integers
    noise_values = np.round(noise_values).astype(int)
    
    # Add results to dataframe
    result_df = result_df.copy()
    result_df['noise'] = noise_values
    result_df['count_noise'] = result_df['count'] + result_df['noise']
    
    # Select only the required columns and return
    final_df = result_df[['values', 'count', 'count_noise', 'noise']].copy()
    return final_df
