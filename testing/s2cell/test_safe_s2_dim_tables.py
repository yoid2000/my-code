import pandas as pd
import sys
import matplotlib.pyplot as plt


def check_one_to_one_mapping(df, col1, col2):
    """
    Check if there's a 1-to-1 mapping between two columns.
    Returns True if 1-to-1, False otherwise, along with details.
    """
    # Group by col1 and check if each value maps to exactly one value in col2
    col1_to_col2 = df.groupby(col1)[col2].nunique()
    col1_violations = col1_to_col2[col1_to_col2 > 1]
    
    # Group by col2 and check if each value maps to exactly one value in col1
    col2_to_col1 = df.groupby(col2)[col1].nunique()
    col2_violations = col2_to_col1[col2_to_col1 > 1]
    
    is_one_to_one = len(col1_violations) == 0 and len(col2_violations) == 0
    
    return is_one_to_one, col1_violations, col2_violations


def create_s2_plot(df):
    """
    Create a plot with lat on x-axis and lon on y-axis, with points connected
    in order of s2 values from smallest to largest.
    """
    try:
        # Get unique lat-lon pairs with their s2 values
        unique_pairs = df[['lat', 'lon', 's2']].drop_duplicates()
        
        # Sort by s2 value
        unique_pairs_sorted = unique_pairs.sort_values('s2')
        
        print(f"Plotting {len(unique_pairs_sorted)} unique lat-lon points")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot points
        plt.scatter(unique_pairs_sorted['lat'], unique_pairs_sorted['lon'], 
                   c='blue', s=0.1, alpha=0.7, label='Lat-Lon Points')
        
        # Connect points with lines in s2 order
        plt.plot(unique_pairs_sorted['lat'], unique_pairs_sorted['lon'], 
                'r-', alpha=0.5, linewidth=0.3, label='S2 Order Connection')
        
        # Add labels and title
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('Lat-Lon Points Connected by S2 Value Order\n(Smallest to Largest S2)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig('safe_s2_dim_tables.png', dpi=1200, bbox_inches='tight')
        plt.close()
        
        print("Plot saved as 'safe_s2_dim_tables.png'")
        print(f"S2 value range: {unique_pairs_sorted['s2'].min()} to {unique_pairs_sorted['s2'].max()}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")


def main():
    try:
        # Read the CSV file
        print("Reading safe_s2_dim_tables.csv...")
        df = pd.read_csv('safe_s2_dim_tables.csv')
        
        print(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns exist
        required_columns = ['lat', 'y', 'lon', 'x', 's2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return
        
        print(f"\nDataframe shape: {df.shape}")
        
        # Check 1-to-1 mapping between lat and y
        print("\n=== Checking 1-to-1 mapping between 'lat' and 'y' ===")
        lat_y_one_to_one, lat_violations, y_violations = check_one_to_one_mapping(df, 'lat', 'y')
        
        if lat_y_one_to_one:
            print("✓ Perfect 1-to-1 mapping between 'lat' and 'y'")
        else:
            print("✗ NOT a 1-to-1 mapping between 'lat' and 'y'")
            print(f"  Number of 'lat' values mapping to multiple 'y' values: {len(lat_violations)}")
            if len(lat_violations) > 0:
                print(f"  Examples: {dict(lat_violations.head())}")
            print(f"  Number of 'y' values mapping to multiple 'lat' values: {len(y_violations)}")
            if len(y_violations) > 0:
                print(f"  Examples: {dict(y_violations.head())}")
        
        # Check 1-to-1 mapping between lon and x
        print("\n=== Checking 1-to-1 mapping between 'lon' and 'x' ===")
        lon_x_one_to_one, lon_violations, x_violations = check_one_to_one_mapping(df, 'lon', 'x')
        
        if lon_x_one_to_one:
            print("✓ Perfect 1-to-1 mapping between 'lon' and 'x'")
        else:
            print("✗ NOT a 1-to-1 mapping between 'lon' and 'x'")
            print(f"  Number of 'lon' values mapping to multiple 'x' values: {len(lon_violations)}")
            if len(lon_violations) > 0:
                print(f"  Examples: {dict(lon_violations.head())}")
            print(f"  Number of 'x' values mapping to multiple 'lon' values: {len(x_violations)}")
            if len(x_violations) > 0:
                print(f"  Examples: {dict(x_violations.head())}")
        
        # Check 1-to-1 mapping between lat-lon pairs and s2
        print("\n=== Checking 1-to-1 mapping between 'lat-lon pairs' and 's2' ===")
        
        # Create a combined lat-lon column for checking
        df['lat_lon_pair'] = df['lat'].astype(str) + ',' + df['lon'].astype(str)
        
        latlon_s2_one_to_one, latlon_violations, s2_violations = check_one_to_one_mapping(df, 'lat_lon_pair', 's2')
        
        if latlon_s2_one_to_one:
            print("✓ Perfect 1-to-1 mapping between 'lat-lon pairs' and 's2'")
        else:
            print("✗ NOT a 1-to-1 mapping between 'lat-lon pairs' and 's2'")
            print(f"  Number of 'lat-lon pairs' mapping to multiple 's2' values: {len(latlon_violations)}")
            if len(latlon_violations) > 0:
                print(f"  Examples: {dict(latlon_violations.head())}")
            print(f"  Number of 's2' values mapping to multiple 'lat-lon pairs': {len(s2_violations)}")
            if len(s2_violations) > 0:
                print(f"  Examples: {dict(s2_violations.head())}")
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"lat ↔ y mapping: {'✓ 1-to-1' if lat_y_one_to_one else '✗ NOT 1-to-1'}")
        print(f"lon ↔ x mapping: {'✓ 1-to-1' if lon_x_one_to_one else '✗ NOT 1-to-1'}")
        print(f"lat-lon pairs ↔ s2 mapping: {'✓ 1-to-1' if latlon_s2_one_to_one else '✗ NOT 1-to-1'}")
        
        # Additional statistics
        print(f"\nUnique values:")
        print(f"  lat: {df['lat'].nunique()} unique values")
        print(f"  y: {df['y'].nunique()} unique values")
        print(f"  lon: {df['lon'].nunique()} unique values")
        print(f"  x: {df['x'].nunique()} unique values")
        print(f"  s2: {df['s2'].nunique()} unique values")
        print(f"  lat-lon pairs: {df['lat_lon_pair'].nunique()} unique pairs")
        
        # Create plot
        print(f"\nCreating plot...")
        create_s2_plot(df)
        
    except FileNotFoundError:
        print("Error: safe_s2_dim_tables.csv not found in current directory")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
