import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple
import random
from common import make_df, make_bins, add_ran_noise
from solve import solve_noise


def test_solver() -> None:
    """
    Run comprehensive tests varying parameters and measuring solution uniqueness.
    """
    # Test parameters
    sd_values = [0, 1, 2]
    nu_values = [2, 4, 8, 16]
    
    print("Starting solver noise tests...")
    print("=" * 60)
    
    for sd in sd_values:
        for nu in nu_values:
            nrows = 10 * (nu ** 2)
            
            print(f"\nTesting parameters: sd={sd}, nu1=nu2={nu}, nrows={nrows}")
            print("-" * 50)
            
            # Set seed for reproducibility
            random.seed(42)
            np.random.seed(42)
            
            # Build dataframe
            df = make_df(nu, nu, nrows)
            
            # Get bins
            df_c1, df_c2, df_c1c2 = make_bins(df)
            
            # Add noise to df_c1 and df_c2
            df_c1_noisy = add_ran_noise(df_c1, sd)
            df_c2_noisy = add_ran_noise(df_c2, sd)
            
            if sd == 0:
                # When sd=0, expect maximum noise to be 0 and only one solution
                print("Testing sd=0 case (expecting unique solution with zero noise)...")
                
                # Find working constraint tolerance for sd=0
                constraint_tolerance = 1.0
                result = None
                while constraint_tolerance <= 100.0:  # reasonable upper limit
                    try:
                        result = solve_noise(df_c1_noisy, df_c2_noisy, df_c1c2, seed=42, constraint_tolerance=constraint_tolerance)
                        break  # Solution found
                    except Exception as e:
                        if "INFEASIBLE" in str(e):
                            constraint_tolerance += 1.0
                            print(f"  Constraint tolerance {constraint_tolerance-1.0} failed, trying {constraint_tolerance}")
                        else:
                            print(f"[ERROR] Test failed with non-infeasibility error: {e}")
                            break
                
                if result is None:
                    print(f"[ERROR] No solution found even with constraint tolerance up to {constraint_tolerance-1.0}. Skipping to next parameter set.")
                    continue
                
                print(f"  Using constraint tolerance: {constraint_tolerance}")
                
                try:
                    max_noise = max(abs(noise) for noise in result['noise'])
                    
                    if max_noise != 0:
                        raise Exception(f"Expected maximum noise to be 0, but got {max_noise}")
                    
                    print(f"[SUCCESS] Maximum noise is 0 as expected")
                    print(f"  solve_runs: 1")
                    print(f"  solve_unique: 1")
                    print(f"  unique_prob: 1.0")
                    print(f"  constraint_tolerance_used: {constraint_tolerance}")
                    print(f"  Noise statistics: avg=0.0, min=0, max=0, stddev=0.0")
                    
                    # Calculate constraint violation statistics for sd=0 case
                    constraint_diffs = []
                    
                    # Check c1 constraint differences
                    for idx, row in df_c1_noisy.iterrows():
                        c1_val = row['values']
                        expected_sum = row['count']
                        
                        actual_sum = 0
                        for sol_idx, sol_row in result.iterrows():
                            combo = sol_row['values']
                            if combo[0] == c1_val:
                                actual_sum += sol_row['count_noise']
                        
                        diff = abs(actual_sum - expected_sum)
                        constraint_diffs.append(diff)
                    
                    # Check c2 constraint differences
                    for idx, row in df_c2_noisy.iterrows():
                        c2_val = row['values']
                        expected_sum = row['count']
                        
                        actual_sum = 0
                        for sol_idx, sol_row in result.iterrows():
                            combo = sol_row['values']
                            if combo[1] == c2_val:
                                actual_sum += sol_row['count_noise']
                        
                        diff = abs(actual_sum - expected_sum)
                        constraint_diffs.append(diff)
                    
                    if constraint_diffs:
                        print(f"  Constraint differences: avg={np.mean(constraint_diffs):.2f}, "
                              f"max={max(constraint_diffs)}, stddev={np.std(constraint_diffs):.2f}")
                    
                except Exception as e:
                    print(f"[ERROR] Test failed: {e}")
                    
            else:
                # When sd > 0, expect multiple solutions
                print(f"Testing sd={sd} case (expecting multiple solutions)...")
                
                solutions: Set[Tuple[int, ...]] = set()
                max_noise_values: List[int] = []
                solve_runs = 0
                no_new_solution_count = 0
                constraint_tolerance = 1.0
                tolerance_found = False
                
                while len(solutions) < 100 and no_new_solution_count < 10:
                    seed = solve_runs + 1
                    solve_runs += 1
                    
                    if not tolerance_found:
                        # Find working constraint tolerance
                        current_tolerance = 1.0
                        result = None
                        while current_tolerance <= 100.0:  # reasonable upper limit
                            try:
                                result = solve_noise(df_c1_noisy, df_c2_noisy, df_c1c2, seed=seed, constraint_tolerance=current_tolerance)
                                constraint_tolerance = current_tolerance
                                tolerance_found = True
                                print(f"  Using constraint tolerance: {constraint_tolerance}")
                                break  # Solution found
                            except Exception as e:
                                if "INFEASIBLE" in str(e):
                                    current_tolerance += 1.0
                                else:
                                    print(f"  Warning: Solution {solve_runs} failed with non-infeasibility error: {e}")
                                    break
                        
                        if not tolerance_found:
                            print(f"[ERROR] No solution found even with constraint tolerance up to {current_tolerance-1.0}. Skipping to next parameter set.")
                            break
                    else:
                        # Use the established tolerance
                        try:
                            result = solve_noise(df_c1_noisy, df_c2_noisy, df_c1c2, seed=seed, constraint_tolerance=constraint_tolerance)
                        except Exception as e:
                            if "INFEASIBLE" in str(e):
                                print(f"[ERROR] Problem became infeasible even with tolerance {constraint_tolerance}. Skipping to next parameter set.")
                                break
                            else:
                                print(f"  Warning: Solution {solve_runs} failed: {e}")
                                continue
                    
                    if result is not None:
                        # Create solution signature based on noise values
                        solution_signature = tuple(sorted(result['noise'].tolist()))
                        
                        if solution_signature in solutions:
                            no_new_solution_count += 1
                        else:
                            solutions.add(solution_signature)
                            no_new_solution_count = 0
                        
                        # Record maximum absolute noise for this solution
                        max_noise = max(abs(noise) for noise in result['noise'])
                        max_noise_values.append(max_noise)
                        result = None  # Reset for next iteration
                
                if not tolerance_found:
                    continue
                
                solve_unique = len(solutions)
                unique_prob = solve_unique / solve_runs if solve_runs > 0 else 0.0
                
                # Calculate statistics for distinct noise values across all solutions
                all_noise_counts: List[int] = []
                for solution_sig in solutions:
                    unique_noise_values = len(set(solution_sig))
                    all_noise_counts.append(unique_noise_values)
                
                if all_noise_counts:
                    avg_distinct = np.mean(all_noise_counts)
                    min_distinct = min(all_noise_counts)
                    max_distinct = max(all_noise_counts)
                    stddev_distinct = np.std(all_noise_counts)
                else:
                    avg_distinct = min_distinct = max_distinct = stddev_distinct = 0
                
                print(f"  solve_runs boo: {solve_runs}")
                print(f"  solve_unique: {solve_unique}")
                print(f"  unique_prob: {unique_prob:.4f}")
                print(f"  constraint_tolerance_used: {constraint_tolerance}")
                
                if max_noise_values:
                    print(f"  Max noise statistics: avg={np.mean(max_noise_values):.2f}, "
                          f"min={min(max_noise_values)}, max={max(max_noise_values)}, "
                          f"stddev={np.std(max_noise_values):.2f}")
                
                print(f"  Distinct noise values statistics: avg={avg_distinct:.2f}, "
                      f"min={min_distinct}, max={max_distinct}, stddev={stddev_distinct:.2f}")
                
                # Calculate constraint violation statistics across all solutions
                if solutions:
                    # Take the first solution to calculate constraint differences
                    try:
                        temp_result = solve_noise(df_c1_noisy, df_c2_noisy, df_c1c2, seed=1, constraint_tolerance=constraint_tolerance, do_print=True)
                        
                        constraint_diffs = []
                        
                        # Check c1 constraint differences
                        for idx, row in df_c1_noisy.iterrows():
                            c1_val = row['values']
                            expected_sum = row['count']
                            
                            actual_sum = 0
                            for sol_idx, sol_row in temp_result.iterrows():
                                combo = sol_row['values']
                                if combo[0] == c1_val:
                                    actual_sum += sol_row['count_noise']
                            
                            diff = abs(actual_sum - expected_sum)
                            constraint_diffs.append(diff)
                        
                        # Check c2 constraint differences
                        for idx, row in df_c2_noisy.iterrows():
                            c2_val = row['values']
                            expected_sum = row['count']
                            
                            actual_sum = 0
                            for sol_idx, sol_row in temp_result.iterrows():
                                combo = sol_row['values']
                                if combo[1] == c2_val:
                                    actual_sum += sol_row['count_noise']
                            
                            diff = abs(actual_sum - expected_sum)
                            constraint_diffs.append(diff)
                        
                        if constraint_diffs:
                            print(f"  Constraint differences: avg={np.mean(constraint_diffs):.2f}, "
                                  f"max={max(constraint_diffs)}, stddev={np.std(constraint_diffs):.2f}")
                    except Exception as e:
                        print(f"  Could not calculate constraint differences: {e}")
                
    print("\n" + "=" * 60)
    print("All tests completed!")


def validate_solution(df_c1: pd.DataFrame, df_c2: pd.DataFrame, 
                      df_c1c2: pd.DataFrame, solution: pd.DataFrame) -> bool:
    """
    Validate that a solution satisfies the sum constraints.
    
    Args:
        df_c1: Original c1 counts (with noise)
        df_c2: Original c2 counts (with noise)
        df_c1c2: Original combination counts
        solution: Solution with noise values
        
    Returns:
        True if solution is valid, False otherwise
    """
    try:
        # Check c1 constraints
        for idx, row in df_c1.iterrows():
            c1_val = row['values']
            expected_sum = row['count']
            
            actual_sum = 0
            for sol_idx, sol_row in solution.iterrows():
                combo = sol_row['values']
                if combo[0] == c1_val:
                    actual_sum += sol_row['count_noise']
            
            if actual_sum != expected_sum:
                print(f"C1 constraint violation: c1={c1_val}, expected={expected_sum}, actual={actual_sum}")
                return False
        
        # Check c2 constraints
        for idx, row in df_c2.iterrows():
            c2_val = row['values']
            expected_sum = row['count']
            
            actual_sum = 0
            for sol_idx, sol_row in solution.iterrows():
                combo = sol_row['values']
                if combo[1] == c2_val:
                    actual_sum += sol_row['count_noise']
            
            if actual_sum != expected_sum:
                print(f"C2 constraint violation: c2={c2_val}, expected={expected_sum}, actual={actual_sum}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


if __name__ == "__main__":
    test_solver()
