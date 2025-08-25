from pyproj import Transformer
import re
import pandas as pd
import os
import json

def epsg2latlon(encoded_str):
    """
    Converts an ESRI-style grid string (e.g., 'CRS3035RES100mN3137650E4201350')
    into latitude and longitude using EPSG:3035 to EPSG:4326 transformation.

    Returns:
        (latitude, longitude) as floats
    """
    # Extract numeric values using regex
    match = re.search(r'N(\d+)E(\d+)', encoded_str)
    if not match:
        raise ValueError("Invalid format: expected 'N<Northing>E<Easting>'")

    northing = int(match.group(1))
    easting = int(match.group(2))

    # Define transformer from EPSG:3035 to EPSG:4326
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    # Transform coordinates
    lon, lat = transformer.transform(easting, northing)

    return lat, lon

def load_esrilatlon_cache(cache_file):
    """
    Load the Gitterzelle to lat/lon mapping cache from JSON file.
    
    Args:
        cache_file: Path to the JSON cache file
        
    Returns:
        dict: Dictionary mapping Gitterzelle strings to {'lat': float, 'lon': float}
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"  Loaded {len(cache)} cached coordinate mappings from {cache_file}")
            return cache
        except Exception as e:
            print(f"  Warning: Error loading cache file {cache_file}: {e}")
            return {}
    else:
        print(f"  Cache file {cache_file} not found - starting with empty cache")
        return {}

def save_esrilatlon_cache(cache, cache_file):
    """
    Save the Gitterzelle to lat/lon mapping cache to JSON file.
    
    Args:
        cache: Dictionary mapping Gitterzelle strings to {'lat': float, 'lon': float}
        cache_file: Path to the JSON cache file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"  Saved {len(cache)} coordinate mappings to {cache_file}")
    except Exception as e:
        print(f"  Warning: Error saving cache file {cache_file}: {e}")

def get_unique_gitterzelle_values(dataframes):
    """
    Get all unique Gitterzelle values from multiple dataframes.
    
    Args:
        dataframes: List of pandas DataFrames
        
    Returns:
        set: Set of unique Gitterzelle values
    """
    unique_values = set()
    for df in dataframes:
        if 'Gitterzelle' in df.columns:
            unique_values.update(df['Gitterzelle'].dropna().unique())
    return unique_values

def process_csv_file(file_path, dataset_name, cache):
    """
    Process a CSV file to add lat/lon columns if they don't exist.
    
    Args:
        file_path: Path to the CSV file
        dataset_name: Name of the dataset for logging
        cache: Dictionary mapping Gitterzelle strings to {'lat': float, 'lon': float}
    
    Returns:
        tuple: (bool indicating if file was modified, updated cache)
    """
    print(f"\nProcessing {dataset_name} dataset: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"  Error: File not found: {file_path}")
        return False, cache
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Check if lat and lon columns already exist
    has_lat = 'lat' in df.columns
    has_lon = 'lon' in df.columns
    
    if has_lat and has_lon:
        print(f"  lat and lon columns already exist in {dataset_name} dataset - skipping conversion")
        return False, cache
    
    # Check if Gitterzelle column exists
    if 'Gitterzelle' not in df.columns:
        print(f"  Error: Gitterzelle column not found in {dataset_name} dataset")
        return False, cache
    
    print(f"  Converting Gitterzelle values to lat/lon coordinates using cache...")
    print(f"  Sample Gitterzelle values: {df['Gitterzelle'].head(3).tolist()}")
    
    # Get unique Gitterzelle values in this dataframe
    unique_gitterzelle = df['Gitterzelle'].dropna().unique()
    print(f"  Found {len(unique_gitterzelle)} unique Gitterzelle values in dataset")
    
    # Check which ones need coordinate lookup
    missing_coords = []
    for gitterzelle in unique_gitterzelle:
        if str(gitterzelle) not in cache:
            missing_coords.append(gitterzelle)
    
    if missing_coords:
        print(f"  Need to look up coordinates for {len(missing_coords)} new Gitterzelle values")
        
        # Perform coordinate transformations for missing values
        for i, gitterzelle in enumerate(missing_coords):
            try:
                lat, lon = epsg2latlon(gitterzelle)
                cache[str(gitterzelle)] = {'lat': lat, 'lon': lon}
            except Exception as e:
                print(f"  Warning: Error converting '{gitterzelle}': {e}")
                cache[str(gitterzelle)] = {'lat': None, 'lon': None}
            
            # Progress indicator for large datasets
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1:,} new coordinates...")
    else:
        print(f"  All Gitterzelle values found in cache - no coordinate lookups needed")
    
    # Create lat and lon columns using the cache
    lats = []
    lons = []
    
    for gitterzelle in df['Gitterzelle']:
        if pd.isna(gitterzelle):
            lats.append(None)
            lons.append(None)
        else:
            coord_data = cache.get(str(gitterzelle), {'lat': None, 'lon': None})
            lats.append(coord_data['lat'])
            lons.append(coord_data['lon'])
    
    # Add lat and lon columns to dataframe
    df['lat'] = lats
    df['lon'] = lons
    
    # Check for any conversion failures
    lat_nulls = df['lat'].isnull().sum()
    lon_nulls = df['lon'].isnull().sum()
    null_coords = max(lat_nulls, lon_nulls)
    if null_coords > 0:
        print(f"  Warning: {null_coords} rows failed coordinate conversion")
    
    print(f"  Coordinate conversion complete:")
    valid_lats = df['lat'].dropna()
    valid_lons = df['lon'].dropna()
    if len(valid_lats) > 0:
        print(f"    Latitude range: {valid_lats.min():.6f} to {valid_lats.max():.6f}")
        print(f"    Longitude range: {valid_lons.min():.6f} to {valid_lons.max():.6f}")
    
    # Write the amended dataframe back to CSV
    print(f"  Writing amended dataframe back to {file_path}...")
    df.to_csv(file_path, index=False)
    print(f"  File written successfully with {len(df.columns)} columns")
    
    return True, cache

def main():
    """
    Main function to process both dense and sparse datasets.
    """
    print("=" * 80)
    print("=== LAT/LON COORDINATE CONVERSION ===")
    print("=" * 80)
    
    # Define file paths
    base_dir = os.path.join("salary_data2")
    dense_file = os.path.join(base_dir, "data_dense.csv")
    sparse_file = os.path.join(base_dir, "data_sparse.csv")
    cache_file = os.path.join(base_dir, "esrilatlon.json")
    
    print(f"Target files:")
    print(f"  Dense dataset: {dense_file}")
    print(f"  Sparse dataset: {sparse_file}")
    print(f"  Cache file: {cache_file}")
    
    # Load existing cache
    cache = load_esrilatlon_cache(cache_file)
    
    # Process both files
    files_processed = []
    
    # Process dense dataset
    dense_modified, cache = process_csv_file(dense_file, "dense", cache)
    if dense_modified:
        files_processed.append("data_dense.csv")
    
    # Process sparse dataset
    sparse_modified, cache = process_csv_file(sparse_file, "sparse", cache)
    if sparse_modified:
        files_processed.append("data_sparse.csv")
    
    # Save updated cache
    if dense_modified or sparse_modified:
        save_esrilatlon_cache(cache, cache_file)
    
    # Summary
    print(f"\n" + "=" * 80)
    print("=== PROCESSING COMPLETE ===")
    print("=" * 80)
    
    if files_processed:
        print(f"Files modified: {len(files_processed)}")
        for file in files_processed:
            print(f"  - {file}")
        print(f"\nLat/lon columns have been added to the modified files.")
        print(f"Coordinate cache contains {len(cache)} mappings.")
    else:
        print("No files were modified - lat/lon columns already exist or errors occurred.")
        print(f"Coordinate cache contains {len(cache)} mappings.")

if __name__ == "__main__":
    main()