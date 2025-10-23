import pandas as pd
import numpy as np
from typing import Tuple
import random


def make_df(nu1: int, nu2: int, nrows: int) -> pd.DataFrame:
    """
    Creates a 2-column dataframe with columns c1 and c2, and nrows rows.
    c1 has nu1 unique integer values, and c2 has nu2 unique integer values.
    Each row has a unique integer value randomly assigned from the unique integers.
    
    Args:
        nu1: Number of unique values for column c1
        nu2: Number of unique values for column c2  
        nrows: Number of rows in the dataframe
        
    Returns:
        DataFrame with columns 'c1' and 'c2'
    """
    # Generate unique values for each column
    c1_values = list(range(nu1))
    c2_values = list(range(nu2))
    
    # Randomly assign values to each row
    c1_data = [random.choice(c1_values) for _ in range(nrows)]
    c2_data = [random.choice(c2_values) for _ in range(nrows)]
    
    df = pd.DataFrame({
        'c1': c1_data,
        'c2': c2_data
    })
    
    return df


def make_bins(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Takes as input the dataframe df from make_df().
    Returns three dataframes with columns 'values' and 'count'.
    
    Args:
        df: Input dataframe with columns 'c1' and 'c2'
        
    Returns:
        Tuple of (df_c1, df_c2, df_c1c2) where:
        - df_c1: One row per unique value in c1 with counts
        - df_c2: One row per unique value in c2 with counts  
        - df_c1c2: One row per unique combination of (c1, c2) with counts
    """
    # Count unique values in c1
    c1_counts = df['c1'].value_counts().reset_index()
    c1_counts.columns = ['values', 'count']
    c1_counts = c1_counts.sort_values('values').reset_index(drop=True)
    
    # Count unique values in c2
    c2_counts = df['c2'].value_counts().reset_index()
    c2_counts.columns = ['values', 'count']
    c2_counts = c2_counts.sort_values('values').reset_index(drop=True)
    
    # Count unique combinations of (c1, c2)
    c1c2_grouped = df.groupby(['c1', 'c2']).size().reset_index(name='count')
    c1c2_grouped['values'] = list(zip(c1c2_grouped['c1'], c1c2_grouped['c2']))
    c1c2_counts = c1c2_grouped[['values', 'count']].copy()
    
    return c1_counts, c2_counts, c1c2_counts


def add_ran_noise(df: pd.DataFrame, sd: float) -> pd.DataFrame:
    """
    Adds random gaussian noise to the count column and rounds to nearest integer.
    
    Args:
        df: Input dataframe with 'count' column
        sd: Standard deviation for gaussian noise
        
    Returns:
        DataFrame with noisy counts rounded to nearest integer
    """
    df_noisy = df.copy()
    
    # Add gaussian noise with mean 0 and given standard deviation
    noise = np.random.normal(0, sd, len(df))
    df_noisy['count'] = np.round(df_noisy['count'] + noise).astype(int)
    
    return df_noisy
