"""
DataFrame Builder - Create synthetic dataframes with specified characteristics

This module provides functions to create dataframes with controllable:
- Column types (categorical or continuous)
- Correlations between columns
- Skewness in distributions
- Number of unique values for categorical columns
"""

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import skewnorm
import warnings
warnings.filterwarnings('ignore')


def df_build(nrows: int=5000, cors: list[list[list[str]]] | None=None, cols: list[dict[str, str | int | float]] | None=None, seed=42) -> tuple[pd.DataFrame, dict]:
    """
    Build a dataframe with specified characteristics.
    
    Parameters:
    -----------
    nrows : int, default 5000
        Number of rows in the dataframe
    ncols : int, default 3
        Number of columns in the dataframe
    cors : list, optional
        List of correlation strengths for column pairs in order:
        Each entry in the list has [[col_index, col_index, ...] strength]
        Where col_index is 0-based index of the column, and strength is one of:
        'weak', 'strong', 'perfect'
    cols : list, optional
        List of column descriptors. Each is a dict with keys:
        - 'type': 'cat', 'con', or 'hybrid
        - 'nuniq': number of unique values (for categorical), or the number of point values (for hybrid)
        - 'skew': 'none', 'weak', 'mid', 'strong'
        - 'bumps': 0 (skewed), 1 (single bump), 2 (two bumps), 3 (three bumps)
        - 'point_fraction': fraction of rows that are point values (for hybrid)
    
    Returns:
    --------
    tuple
        (pd.DataFrame, dict) where dict contains all parameters used to create the dataframe
    """
    np.random.seed(seed)
    # Set defaults
    if cols is None:
        cols = [{'type': 'con', 'skew': 'none'}]
    if cors is None:
        cors = []
    
    ncols = len(cols)
    
    # Create initial dataframe with float columns
    df_data = {}
    
    for i in range(ncols):
        col_name = f'c{i}'
        col_spec = cols[i]
        skew_level = col_spec.get('skew', 'none')
        bumps = col_spec.get('bumps', 0)
        
        # Generate values based on skew and bumps
        if skew_level == 'none':
            values = np.random.uniform(0.0, 1.0, nrows)
        else:
            if bumps == 0:
                # Original skewed distribution
                if skew_level == 'weak':
                    alpha, beta = 0.8, 2.0
                elif skew_level == 'mid':
                    alpha, beta = 0.5, 3.0
                elif skew_level == 'strong':
                    alpha, beta = 0.4, 3.0
                else:
                    alpha, beta = 1.0, 1.0  # uniform fallback
                
                values = np.random.beta(alpha, beta, nrows)
                
            elif bumps == 1:
                # Single bump in the middle using normal distribution
                # (higher skew = narrower bump with steeper curves)
                if skew_level == 'weak':
                    std = 0.14  # wider bump
                elif skew_level == 'mid':
                    std = 0.1  # narrower bump
                elif skew_level == 'strong':
                    std = 0.06  # very narrow, steep bump
                else:
                    std = 0.15  # default bump
                
                # Generate normal distribution centered at 0.5
                values = np.random.normal(0.5, std, nrows)
                # Clip to [0,1] range
                values = np.clip(values, 0.0, 1.0)
                
            elif bumps == 2:
                # Two bumps - mixture of two beta distributions
                if skew_level == 'weak':
                    # Wide bumps
                    alpha1 = beta1 = 2.5
                    alpha2 = beta2 = 2.5
                elif skew_level == 'mid':
                    # Narrower bumps (increased from 4.0)
                    alpha1 = beta1 = 7.0
                    alpha2 = beta2 = 7.0
                elif skew_level == 'strong':
                    # Very narrow bumps (increased from 6.0)
                    alpha1 = beta1 = 12.0
                    alpha2 = beta2 = 12.0
                else:
                    alpha1 = beta1 = alpha2 = beta2 = 2.0
                
                # Create two populations at different locations
                n1 = nrows // 2
                n2 = nrows - n1
                
                # First bump around 0.25
                vals1 = np.random.beta(alpha1, beta1, n1) * 0.4 + 0.05
                # Second bump around 0.75
                vals2 = np.random.beta(alpha2, beta2, n2) * 0.4 + 0.55
                
                values = np.concatenate([vals1, vals2])
                
            elif bumps == 3:
                # Three bumps - mixture of three beta distributions
                if skew_level == 'weak':
                    # Wide bumps
                    alpha = beta = 2.0
                elif skew_level == 'mid':
                    # Narrower bumps (increased from 3.5)
                    alpha = beta = 6.0
                elif skew_level == 'strong':
                    # Very narrow bumps (increased from 5.0)
                    alpha = beta = 10.0
                else:
                    alpha = beta = 2.0
                
                # Create three populations
                n1 = nrows // 3
                n2 = nrows // 3
                n3 = nrows - n1 - n2
                
                # Three bumps at 0.17, 0.5, 0.83
                vals1 = np.random.beta(alpha, beta, n1) * 0.25 + 0.05
                vals2 = np.random.beta(alpha, beta, n2) * 0.25 + 0.375
                vals3 = np.random.beta(alpha, beta, n3) * 0.25 + 0.705
                
                values = np.concatenate([vals1, vals2, vals3])
            
            else:
                # Fallback to uniform if bumps value is invalid
                values = np.random.uniform(0.0, 1.0, nrows)
        
        # Sort values low to high initially
        values.sort()
        df_data[col_name] = values
    
    df = pd.DataFrame(df_data)
    
    # Handle correlations
    correlated_columns = set()
    
    for cor_entry in cors:
        # Handle different possible structures of cor_entry
        col_indices = cor_entry[0]
        strength = cor_entry[1]
        
        # Add to correlated columns set
        correlated_columns.update(col_indices)
        
        # Determine swap percentage based on strength
        if strength == 'perfect':
            swap_pct = 0.0
        elif strength == 'strong':
            swap_pct = 0.2
        elif strength == 'weak':
            swap_pct = 0.8
        else:
            swap_pct = 0.0
        
        # For each column in the group, randomly swap positions
        for col_idx in col_indices:
            col_name = f'c{col_idx}'
            if swap_pct > 0:
                n_swaps = int(nrows * swap_pct)
                swap_indices = np.random.choice(nrows, n_swaps, replace=False)
                # Randomly shuffle just these positions
                swap_values = df[col_name].iloc[swap_indices].values
                np.random.shuffle(swap_values)
                df.loc[swap_indices, col_name] = swap_values
        
        # Shuffle rows for the entire group together
        group_cols = [f'c{i}' for i in col_indices]
        shuffle_indices = np.random.permutation(nrows)
        df[group_cols] = df[group_cols].iloc[shuffle_indices].values
    
    # Randomly shuffle uncorrelated columns
    for i in range(ncols):
        if i not in correlated_columns:
            col_name = f'c{i}'
            df[col_name] = np.random.permutation(df[col_name].values)
    
    # Process column types
    for i in range(ncols):
        col_name = f'c{i}'
        col_spec = cols[i]
        col_type = col_spec.get('type', 'con')
        
        if col_type == 'cat':
            nuniq = int(col_spec.get('nuniq', 10))
            # Create histogram bins
            bin_width = 1.0 / nuniq
            # Assign bin names as binxx_yyyy format (xx = zero-padded number, yyyy = 4 random letters)
            bin_labels = [f'bin{j:02d}_{''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=4))}' for j in range(nuniq)]
            
            # Assign bin labels based on value ranges
            values = df[col_name].values
            bin_indices = np.floor(values / bin_width).astype(int)
            # Handle edge case where value = 1.0
            bin_indices = np.clip(bin_indices, 0, nuniq - 1)
            
            df[col_name] = [bin_labels[idx] for idx in bin_indices]
            df[col_name] = df[col_name].astype(str)
            
        elif col_type == 'hybrid':
            nuniq = int(col_spec.get('nuniq', 5))
            point_fraction = float(col_spec.get('point_fraction', 0.1))
            
            # Generate nuniq random point values
            point_values = np.sort(np.random.uniform(0.0, 1.0, nuniq))
            
            # Select random indices for point values
            num_points = int(point_fraction * nrows)
            selected_indices = np.random.choice(nrows, num_points, replace=False)
            selected_indices.sort()  # Sort to assign lowest values to lowest indices
            
            # Assign point values
            points_per_value = num_points // nuniq
            remainder = num_points % nuniq
            
            start_idx = 0
            for j in range(nuniq):
                # Calculate how many indices get this point value
                count = points_per_value + (1 if j < remainder else 0)
                end_idx = start_idx + count
                
                if end_idx > start_idx:
                    indices_for_this_value = selected_indices[start_idx:end_idx]
                    df.loc[indices_for_this_value, col_name] = point_values[j]
                
                start_idx = end_idx
    
    # Create parameter description dictionary
    params_dict = {
        'nrows': nrows,
        'ncols': ncols,
        'cors': cors,
        'cols': cols,
        'column_names': [f'c{i}' for i in range(ncols)],
        'column_details': {}
    }
    
    # Add detailed column information
    for i in range(ncols):
        col_name = f'c{i}'
        col_spec = cols[i]
        params_dict['column_details'][col_name] = {
            'type': col_spec.get('type', 'con'),
            'skew': col_spec.get('skew', 'none'),
            'bumps': col_spec.get('bumps', 0),
        }
        if col_spec.get('type') == 'cat':
            params_dict['column_details'][col_name]['nuniq'] = col_spec.get('nuniq', 10)
        elif col_spec.get('type') == 'hybrid':
            params_dict['column_details'][col_name]['nuniq'] = col_spec.get('nuniq', 5)
            params_dict['column_details'][col_name]['point_fraction'] = col_spec.get('point_fraction', 0.1)
    
    return df, params_dict

def name_from_params(params_dict: dict) -> str:
    """
    Generate a compact, unique string identifier from df_build parameters.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary returned by df_build() containing build parameters
        
    Returns:
    --------
    str
        Compact string uniquely identifying the parameter set
    """
    # Extract basic info
    nrows = params_dict.get('nrows', 0)
    ncols = params_dict.get('ncols', 0)
    
    # Process column types and properties
    col_details = params_dict.get('column_details', {})
    type_str = ""
    skew_str = ""
    bumps_str = ""
    nuniq_str = ""
    point_frac_str = ""
    
    for i in range(ncols):
        col_name = f'c{i}'
        if col_name in col_details:
            col_info = col_details[col_name]
            col_type = col_info.get('type', 'con')
            col_skew = col_info.get('skew', 'none')
            col_bumps = col_info.get('bumps', 0)
            
            # Type abbreviation
            if col_type == 'cat':
                type_str += 'C'
            elif col_type == 'hybrid':
                type_str += 'H'
            else:
                type_str += 'N'
            
            # Skew abbreviation
            skew_map = {'none': '0', 'weak': '1', 'mid': '2', 'strong': '3'}
            skew_str += skew_map.get(col_skew, '0')
            
            # Bumps abbreviation
            bumps_str += str(col_bumps)
            
            # Nuniq for categorical and hybrid columns
            if col_type in ['cat', 'hybrid']:
                nuniq = col_info.get('nuniq', 10)
                nuniq_str += f"{nuniq}"
            else:
                nuniq_str += ""
                
            # Point fraction for hybrid columns
            if col_type == 'hybrid':
                point_frac = col_info.get('point_fraction', 0.1)
                # Convert to percentage and round to avoid floating point issues
                point_frac_pct = int(round(point_frac * 100))
                point_frac_str += f"{point_frac_pct}"
            else:
                point_frac_str += "-"
    
    # Process correlations
    cors = params_dict.get('cors', [])
    cor_str = ""
    if cors:
        for cor_entry in cors:
            col_indices = cor_entry[0]
            strength = cor_entry[1]
            
            # Create correlation descriptor: indices_strength
            indices_part = "".join(map(str, sorted(col_indices)))
            
            strength_map = {'weak': 'w', 'strong': 's', 'perfect': 'p'}
            strength_part = strength_map.get(strength, 'w')
            
            cor_str += f"{indices_part}{strength_part}"
        
        # If multiple correlation groups, separate with underscore
        if len(cors) > 1:
            cor_str = "_".join([f"{cor_entry[0]}{cor_entry[1][0]}" for cor_entry in cors])
    else:
        cor_str = "none"
    
    # Combine into compact format
    name_parts = [
        f"r{nrows}",
        f"c{ncols}",
        f"t{type_str}",
        f"s{skew_str}",
        f"b{bumps_str}",
        f"u{nuniq_str}",
        f"cor{cor_str}"
    ]
    
    # Add point fraction info if any hybrid columns exist
    if 'H' in type_str:
        name_parts.append(f"pf{point_frac_str}")
    
    return "_".join(name_parts)

def df_describe(df):
    """
    Generate a comprehensive description of a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to describe
        
    Returns:
    --------
    tuple
        (str, dict) where str is the text description and dict contains all the information
    """
    description = []
    description_dict = {}
    
    # Basic shape information
    description.append(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    description_dict['shape'] = {'rows': int(df.shape[0]), 'columns': int(df.shape[1])}
    description_dict['columns'] = {}
    
    # Column information
    description.append("Column Information:")
    description.append("-" * 50)
    
    for col in df.columns:
        col_type = "Categorical" if df[col].dtype == 'object' else "Continuous"
        description.append(f"\n{col} ({col_type}):")
        
        col_info = {'type': col_type.lower(), 'dtype': str(df[col].dtype)}
        
        if df[col].dtype == 'object':
            # Categorical column analysis
            nuniq = df[col].nunique()
            value_counts = df[col].value_counts()
            description.append(f"  Unique values: {nuniq}")
            description.append(f"  Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
            description.append(f"  Least common: {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)")
            
            col_info['unique_values'] = int(nuniq)
            col_info['most_common'] = {'value': value_counts.index[0], 'count': int(value_counts.iloc[0])}
            col_info['least_common'] = {'value': value_counts.index[-1], 'count': int(value_counts.iloc[-1])}
            
            # Add count for each category
            description.append(f"  Category counts:")
            col_info['category_counts'] = {}
            for category, count in value_counts.items():
                description.append(f"    {category}: {count}")
                col_info['category_counts'][category] = int(count)
            
            # Skewness analysis for categorical
            max_count = value_counts.max()
            min_count = value_counts.min()
            skew_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if skew_ratio < 1.5:
                skew_desc = "Low skew (relatively uniform)"
            elif skew_ratio < 3:
                skew_desc = "Moderate skew"
            else:
                skew_desc = "High skew (concentrated distribution)"
            
            description.append(f"  Distribution: {skew_desc} (ratio: {skew_ratio:.2f})")
            col_info['distribution'] = {'description': skew_desc, 'skew_ratio': float(skew_ratio)}
            
        else:
            # Continuous column analysis
            stats = df[col].describe()
            description.append(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            description.append(f"  Mean: {stats['mean']:.3f}")
            description.append(f"  Std: {stats['std']:.3f}")
            
            col_info['range'] = {'min': float(stats['min']), 'max': float(stats['max'])}
            col_info['mean'] = float(stats['mean'])
            col_info['std'] = float(stats['std'])
            col_info['quartiles'] = {
                '25%': float(stats['25%']),
                '50%': float(stats['50%']),
                '75%': float(stats['75%'])
            }
            
            # Skewness
            skewness = df[col].skew()
            if abs(skewness) < 0.5:
                skew_desc = "Low skew (roughly normal)"
            elif abs(skewness) < 1:
                skew_desc = "Moderate skew"
            else:
                skew_desc = "High skew"
            description.append(f"  Skewness: {skewness:.3f} ({skew_desc})")
            col_info['skewness'] = {'value': float(skewness), 'description': skew_desc}
        
        description_dict['columns'][col] = col_info
    
    # Correlation analysis
    description.append("\n\nCorrelation Analysis:")
    description.append("-" * 50)
    description_dict['correlations'] = {}
    
    # For continuous columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        description.append("\nContinuous column correlations:")
        description_dict['correlations']['continuous'] = {}
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Only show upper triangle
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) < 0.2:
                        strength = "weak"
                    elif abs(corr_val) < 0.6:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    description.append(f"  {col1} - {col2}: {corr_val:.3f} ({strength})")
                    description_dict['correlations']['continuous'][f"{col1}-{col2}"] = {
                        'value': float(corr_val), 'strength': strength
                    }
    
    # Categorical-categorical associations
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) > 1:
        description.append("\nCategorical-Categorical associations:")
        description_dict['correlations']['categorical'] = {}
        
        for i, cat_col1 in enumerate(cat_cols):
            for j, cat_col2 in enumerate(cat_cols):
                if i < j:  # Only show upper triangle
                    # Calculate Cramér's V using contingency table
                    confusion_matrix = pd.crosstab(df[cat_col1], df[cat_col2])
                    
                    # Calculate chi-squared statistic
                    row_totals = confusion_matrix.sum(axis=1)
                    col_totals = confusion_matrix.sum(axis=0)
                    total = confusion_matrix.sum().sum()
                    
                    expected = np.outer(row_totals, col_totals) / total
                    chi2 = ((confusion_matrix - expected) ** 2 / expected).sum().sum()
                    
                    # Cramér's V calculation
                    n = total
                    min_dim = min(confusion_matrix.shape[0] - 1, confusion_matrix.shape[1] - 1)
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    
                    if cramers_v < 0.1:
                        strength = "weak"
                    elif cramers_v < 0.3:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    description.append(f"  {cat_col1} - {cat_col2}: V = {cramers_v:.3f} ({strength})")
                    description_dict['correlations']['categorical'][f"{cat_col1}-{cat_col2}"] = {
                        'cramers_v': float(cramers_v), 'strength': strength
                    }

    # Mixed correlations (categorical vs continuous)
    if cat_cols and numeric_cols:
        description.append("\nCategorical-Continuous associations:")
        description_dict['correlations']['mixed'] = {}
        
        for cat_col in cat_cols:
            for num_col in numeric_cols:
                # Use eta-squared (correlation ratio) for cat-num association
                groups = [df[df[cat_col] == cat][num_col].values for cat in df[cat_col].unique()]
                # Simple measure: variance between groups / total variance
                overall_mean = df[num_col].mean()
                between_var = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in groups if len(group) > 0)
                total_var = df[num_col].var() * len(df)
                
                if total_var > 0:
                    eta_squared = between_var / total_var
                    if eta_squared < 0.01:
                        strength = "weak"
                    elif eta_squared < 0.06:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    description.append(f"  {cat_col} - {num_col}: eta-sq = {eta_squared:.3f} ({strength})")
                    description_dict['correlations']['mixed'][f"{cat_col}-{num_col}"] = {
                        'eta_squared': float(eta_squared), 'strength': strength
                    }
    
    return "\n".join(description), description_dict
