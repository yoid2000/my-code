from geopy.geocoders import Nominatim
import pandas as pd
import os
import time
import json
from collections import defaultdict

def latlon_to_zip(lat, lon):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.reverse((lat, lon), exactly_one=True)
    return location.raw.get('address', {}).get('postcode', None)

def get_unique_latlon_pairs(df_dense, df_sparse):
    """
    Get all unique lat/lon pairs from both datasets.
    
    Returns:
        set: Set of (lat, lon) tuples rounded to 6 decimal places
    """
    print("  Finding unique lat/lon pairs...")
    
    # Combine lat/lon pairs from both datasets, rounding to 6 decimal places
    dense_pairs = set((round(lat, 6), round(lon, 6)) for lat, lon in zip(df_dense['lat'], df_dense['lon']) if pd.notna(lat) and pd.notna(lon))
    sparse_pairs = set((round(lat, 6), round(lon, 6)) for lat, lon in zip(df_sparse['lat'], df_sparse['lon']) if pd.notna(lat) and pd.notna(lon))
    
    # Get unique pairs across both datasets
    all_pairs = dense_pairs.union(sparse_pairs)
    
    print(f"    Dense dataset: {len(dense_pairs)} unique lat/lon pairs")
    print(f"    Sparse dataset: {len(sparse_pairs)} unique lat/lon pairs")
    print(f"    Combined unique pairs: {len(all_pairs)}")
    print(f"    Valid pairs (no NaN): {len(all_pairs)}")
    
    return all_pairs

def load_latlon_mapping(mapping_file):
    """
    Load lat/lon to zip mapping from JSON file if it exists.
    
    Args:
        mapping_file: Path to JSON mapping file
        
    Returns:
        dict: Mapping from (lat, lon) tuples to zip codes, empty if file doesn't exist
    """
    if not os.path.exists(mapping_file):
        print(f"  No existing mapping file found: {mapping_file}")
        return {}
    
    try:
        with open(mapping_file, 'r') as f:
            json_data = json.load(f)
        
        # Convert string keys back to (lat, lon) tuples
        latlon_to_zip_map = {}
        for key, value in json_data.items():
            # Parse key format: "lat,lon"
            lat_str, lon_str = key.split(',')
            lat, lon = float(lat_str), float(lon_str)
            latlon_to_zip_map[(lat, lon)] = value
        
        print(f"  Loaded existing mapping with {len(latlon_to_zip_map)} lat/lon pairs from: {mapping_file}")
        return latlon_to_zip_map
        
    except Exception as e:
        print(f"  Error loading mapping file {mapping_file}: {e}")
        return {}

def save_latlon_mapping(latlon_to_zip_map, mapping_file):
    """
    Save lat/lon to zip mapping to JSON file.
    
    Args:
        latlon_to_zip_map: Mapping from (lat, lon) tuples to zip codes
        mapping_file: Path to JSON mapping file
    """
    try:
        # Convert (lat, lon) tuples to string keys for JSON serialization
        json_data = {}
        for (lat, lon), zip_code in latlon_to_zip_map.items():
            key = f"{lat},{lon}"
            json_data[key] = zip_code
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        
        with open(mapping_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"  Saved mapping with {len(latlon_to_zip_map)} lat/lon pairs to: {mapping_file}")
        
    except Exception as e:
        print(f"  Error saving mapping file {mapping_file}: {e}")

def create_latlon_to_zip_mapping(unique_pairs, existing_mapping=None, mapping_file=None, max_retries=10):
    """
    Create a mapping from lat/lon pairs to zip codes with automatic retries for failed lookups.
    
    Args:
        unique_pairs: Set of (lat, lon) tuples
        existing_mapping: Existing mapping to extend (optional)
        mapping_file: Path to save mapping file (optional)
        max_retries: Maximum number of retry passes (default: 10)
        
    Returns:
        dict: Mapping from (lat, lon) to zip code
    """
    # Start with existing mapping if provided
    latlon_to_zip_map = existing_mapping.copy() if existing_mapping else {}
    
    # Find pairs that need lookup
    pairs_to_lookup = unique_pairs - set(latlon_to_zip_map.keys())
    
    if len(pairs_to_lookup) == 0:
        print(f"  All {len(unique_pairs)} lat/lon pairs already in existing mapping - no API calls needed!")
        return latlon_to_zip_map
    
    print(f"  Need to lookup {len(pairs_to_lookup)} new locations (out of {len(unique_pairs)} total)")
    print(f"  Using existing mapping for {len(unique_pairs) - len(pairs_to_lookup)} locations")
    
    retry_attempt = 0
    current_pairs_to_lookup = list(pairs_to_lookup)
    
    while current_pairs_to_lookup and retry_attempt < max_retries:
        retry_attempt += 1
        failed_pairs = []
        successful_in_pass = 0
        
        print(f"\n  === PASS {retry_attempt}: Processing {len(current_pairs_to_lookup)} locations ===")
        
        for i, (lat, lon) in enumerate(current_pairs_to_lookup):
            try:
                zip_code = latlon_to_zip(lat, lon)
                latlon_to_zip_map[(lat, lon)] = zip_code
                
                if zip_code is None:
                    failed_pairs.append((lat, lon))
                else:
                    successful_in_pass += 1
                
                # Progress indicator and rate limiting - save every 10 requests
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i + 1}/{len(current_pairs_to_lookup)} locations in pass {retry_attempt}...")
                    
                    # Save progress every 10 requests
                    if mapping_file:
                        save_latlon_mapping(latlon_to_zip_map, mapping_file)
                    
                    time.sleep(1)  # Rate limiting to be respectful to the geocoding service
                    
            except Exception as e:
                print(f"    Warning: Failed to get zip code for ({lat:.6f}, {lon:.6f}): {e}")
                latlon_to_zip_map[(lat, lon)] = None
                failed_pairs.append((lat, lon))
                time.sleep(1)  # Brief pause after errors
        
        # Report pass results
        failed_in_pass = len(failed_pairs)
        print(f"    Pass {retry_attempt} complete: {successful_in_pass} successful, {failed_in_pass} failed")
        
        # Save progress after each pass
        if mapping_file:
            save_latlon_mapping(latlon_to_zip_map, mapping_file)
        
        # Prepare for next pass if there are failures
        current_pairs_to_lookup = failed_pairs
        
        if current_pairs_to_lookup:
            if retry_attempt < max_retries:
                wait_time = min(2 ** retry_attempt, 30)  # Exponential backoff, max 30 seconds
                print(f"    Waiting {wait_time} seconds before retry pass {retry_attempt + 1}...")
                time.sleep(wait_time)
            else:
                print(f"    Reached maximum retries ({max_retries}). {len(current_pairs_to_lookup)} locations still failed.")
        else:
            print(f"    All locations successfully processed after {retry_attempt} pass(es)!")
    
    # Final statistics
    total_processed = len(pairs_to_lookup)
    total_successful = total_processed - len(current_pairs_to_lookup)
    total_in_map = len(unique_pairs) - len(current_pairs_to_lookup)
    
    print(f"\n  === FINAL RESULTS ===")
    print(f"    New lookups: {total_successful}/{total_processed} successful")
    print(f"    Total mapping: {total_in_map}/{len(unique_pairs)} successful lookups")
    if current_pairs_to_lookup:
        print(f"    Remaining failed locations: {len(current_pairs_to_lookup)}")
    
    # Save final mapping
    if mapping_file:
        save_latlon_mapping(latlon_to_zip_map, mapping_file)
    
    return latlon_to_zip_map

def add_zip_column(df, dataset_name, latlon_to_zip_map):
    """
    Add zip column to dataframe using the lat/lon to zip mapping.
    
    Args:
        df: DataFrame to modify
        dataset_name: Name of the dataset for logging
        latlon_to_zip_map: Mapping from (lat, lon) to zip code
        
    Returns:
        bool: True if column was added, False if already exists
    """
    print(f"  Adding zip column to {dataset_name} dataset...")
    
    # Check if zip column already exists
    if 'zip' in df.columns:
        print(f"    zip column already exists in {dataset_name} dataset - skipping")
        return False
    
    # Create zip column using the mapping
    zip_codes = []
    missing_coords = 0
    
    for _, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        
        if pd.isna(lat) or pd.isna(lon):
            zip_codes.append(None)
            missing_coords += 1
        else:
            # Round lat/lon to 6 decimal places for lookup
            rounded_lat, rounded_lon = round(lat, 6), round(lon, 6)
            zip_code = latlon_to_zip_map.get((rounded_lat, rounded_lon), None)
            zip_codes.append(zip_code)
    
    df['zip'] = zip_codes
    
    # Report statistics
    total_rows = len(df)
    non_null_zips = sum(1 for z in zip_codes if z is not None)
    
    print(f"    {dataset_name} dataset zip column statistics:")
    print(f"      Total rows: {total_rows:,}")
    print(f"      Missing coordinates: {missing_coords:,}")
    print(f"      Valid zip codes: {non_null_zips:,}")
    print(f"      Missing zip codes: {total_rows - non_null_zips:,}")
    
    return True

def find_missing_zip_locations(df_dense, df_sparse, latlon_to_zip_map):
    """
    Find lat/lon pairs that have missing zip codes in either the mapping or the CSV files.
    
    Args:
        df_dense: Dense dataset DataFrame
        df_sparse: Sparse dataset DataFrame
        latlon_to_zip_map: Existing mapping from (lat, lon) to zip codes
        
    Returns:
        set: Set of (lat, lon) tuples that need zip code lookup
    """
    print("  Finding locations with missing zip codes...")
    
    missing_pairs = set()
    
    # Check mapping file for null zip codes
    mapping_missing = 0
    for (lat, lon), zip_code in latlon_to_zip_map.items():
        if zip_code is None:
            missing_pairs.add((lat, lon))
            mapping_missing += 1
    
    print(f"    Found {mapping_missing} locations with missing zip codes in mapping file")
    
    # Check CSV files for missing zip codes (if zip columns exist)
    csv_missing = 0
    for df, name in [(df_dense, 'dense'), (df_sparse, 'sparse')]:
        if 'zip' in df.columns:
            for _, row in df.iterrows():
                lat, lon = row['lat'], row['lon']
                if pd.notna(lat) and pd.notna(lon) and pd.isna(row['zip']):
                    rounded_lat, rounded_lon = round(lat, 6), round(lon, 6)
                    missing_pairs.add((rounded_lat, rounded_lon))
                    csv_missing += 1
    
    print(f"    Found {csv_missing} locations with missing zip codes in CSV files")
    print(f"    Total unique locations needing zip code lookup: {len(missing_pairs)}")
    
    return missing_pairs

def retry_missing_zip_lookups(missing_pairs, latlon_to_zip_map, mapping_file=None, max_retries=10):
    """
    Retry zip code lookups for locations with missing zip codes using multiple passes.
    
    Args:
        missing_pairs: Set of (lat, lon) tuples needing lookup
        latlon_to_zip_map: Existing mapping to update
        mapping_file: Path to save updated mapping (optional)
        max_retries: Maximum number of retry passes (default: 10)
        
    Returns:
        dict: Updated mapping with new zip code lookups
    """
    if len(missing_pairs) == 0:
        print("  No missing zip codes to retry - all locations have valid zip codes!")
        return latlon_to_zip_map
    
    print(f"  Retrying zip code lookups for {len(missing_pairs)} locations with missing data...")
    
    retry_attempt = 0
    current_pairs_to_retry = list(missing_pairs)
    
    while current_pairs_to_retry and retry_attempt < max_retries:
        retry_attempt += 1
        failed_pairs = []
        successful_in_pass = 0
        
        print(f"\n  === RETRY PASS {retry_attempt}: Processing {len(current_pairs_to_retry)} failed locations ===")
        
        for i, (lat, lon) in enumerate(current_pairs_to_retry):
            try:
                zip_code = latlon_to_zip(lat, lon)
                latlon_to_zip_map[(lat, lon)] = zip_code
                
                if zip_code is None:
                    failed_pairs.append((lat, lon))
                else:
                    successful_in_pass += 1
                
                # Progress indicator and periodic saving
                if (i + 1) % 10 == 0:
                    print(f"    Retried {i + 1}/{len(current_pairs_to_retry)} locations in pass {retry_attempt}...")
                    
                    # Save progress every 10 requests
                    if mapping_file:
                        save_latlon_mapping(latlon_to_zip_map, mapping_file)
                    
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                print(f"    Warning: Failed to retry zip code for ({lat:.6f}, {lon:.6f}): {e}")
                latlon_to_zip_map[(lat, lon)] = None
                failed_pairs.append((lat, lon))
                time.sleep(1)  # Brief pause after errors
        
        # Report pass results
        failed_in_pass = len(failed_pairs)
        print(f"    Retry pass {retry_attempt} complete: {successful_in_pass} successful, {failed_in_pass} still failed")
        
        # Save progress after each pass
        if mapping_file:
            save_latlon_mapping(latlon_to_zip_map, mapping_file)
        
        # Prepare for next pass if there are failures
        current_pairs_to_retry = failed_pairs
        
        if current_pairs_to_retry:
            if retry_attempt < max_retries:
                wait_time = min(2 ** retry_attempt, 30)  # Exponential backoff, max 30 seconds
                print(f"    Waiting {wait_time} seconds before retry pass {retry_attempt + 1}...")
                time.sleep(wait_time)
            else:
                print(f"    Reached maximum retry attempts ({max_retries}). {len(current_pairs_to_retry)} locations still failed.")
        else:
            print(f"    All missing locations successfully processed after {retry_attempt} retry pass(es)!")
    
    # Final retry statistics
    total_retried = len(missing_pairs)
    total_successful = total_retried - len(current_pairs_to_retry)
    
    print(f"\n  === RETRY RESULTS ===")
    print(f"    Retry attempts: {total_successful}/{total_retried} successful")
    if current_pairs_to_retry:
        print(f"    Still failed after retries: {len(current_pairs_to_retry)}")
    
    return latlon_to_zip_map

def get_completion_status(latlon_to_zip_map):
    """
    Get completion status of zip code lookups.
    
    Args:
        latlon_to_zip_map: Mapping from (lat, lon) tuples to zip codes
        
    Returns:
        tuple: (total_locations, successful_lookups, failed_locations)
    """
    total_locations = len(latlon_to_zip_map)
    successful_lookups = sum(1 for v in latlon_to_zip_map.values() if v is not None)
    failed_locations = total_locations - successful_lookups
    
    return total_locations, successful_lookups, failed_locations

def process_datasets():
    """
    Main function to process both datasets and add zip codes.
    """
    print("=" * 80)
    print("=== LAT/LON TO ZIP CODE CONVERSION ===")
    print("=" * 80)
    
    # Define file paths
    data_dir = "salary_data2"
    dense_file = os.path.join(data_dir, "data_dense.csv")
    sparse_file = os.path.join(data_dir, "data_sparse.csv")
    mapping_file = os.path.join(data_dir, "latlon_map.json")
    
    print(f"Target files:")
    print(f"  Dense dataset: {dense_file}")
    print(f"  Sparse dataset: {sparse_file}")
    print(f"  Mapping file: {mapping_file}")
    
    # Check if files exist
    for file_path in [dense_file, sparse_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Read datasets
    print(f"\n" + "=" * 60)
    print("=== READING DATASETS ===")
    print("=" * 60)
    
    print(f"Reading dense dataset...")
    df_dense = pd.read_csv(dense_file)
    print(f"  Dense dataset: {df_dense.shape[0]:,} rows × {df_dense.shape[1]} columns")
    
    print(f"Reading sparse dataset...")
    df_sparse = pd.read_csv(sparse_file)
    print(f"  Sparse dataset: {df_sparse.shape[0]:,} rows × {df_sparse.shape[1]} columns")
    
    # Check if required columns exist
    required_cols = ['lat', 'lon']
    for df, name in [(df_dense, 'dense'), (df_sparse, 'sparse')]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns in {name} dataset: {missing_cols}")
            return
    
    # Load existing mapping if available
    print(f"\n" + "=" * 60)
    print("=== LOADING EXISTING MAPPING ===")
    print("=" * 60)
    
    existing_mapping = load_latlon_mapping(mapping_file)
    
    # Check for missing zip codes and retry lookups if needed
    print(f"\n" + "=" * 60)
    print("=== CHECKING FOR MISSING ZIP CODES ===")
    print("=" * 60)
    
    missing_pairs = find_missing_zip_locations(df_dense, df_sparse, existing_mapping)
    
    if len(missing_pairs) > 0:
        print(f"\n" + "=" * 60)
        print("=== RETRYING MISSING ZIP CODE LOOKUPS ===")
        print("=" * 60)
        
        existing_mapping = retry_missing_zip_lookups(missing_pairs, existing_mapping, mapping_file)
    
    # Check if zip columns already exist
    dense_has_zip = 'zip' in df_dense.columns
    sparse_has_zip = 'zip' in df_sparse.columns
    
    if dense_has_zip and sparse_has_zip and len(missing_pairs) == 0:
        print(f"\nzip columns already exist in both datasets with no missing values - skipping conversion")
        return
    
    # Get unique lat/lon pairs for any new locations
    print(f"\n" + "=" * 60)
    print("=== FINDING UNIQUE LAT/LON PAIRS ===")
    print("=" * 60)
    
    unique_pairs = get_unique_latlon_pairs(df_dense, df_sparse)
    
    if len(unique_pairs) == 0:
        print("No valid lat/lon pairs found - cannot proceed")
        return
    
    # Create lat/lon to zip mapping for any new locations
    print(f"\n" + "=" * 60)
    print("=== CREATING ZIP CODE MAPPING ===")
    print("=" * 60)

    # Create or extend the mapping with initial pass
    latlon_to_zip_map = create_latlon_to_zip_mapping(unique_pairs, existing_mapping, mapping_file)
    
    # Continue retrying until all lookups succeed or max attempts reached
    max_overall_retries = 20
    overall_attempt = 0
    
    while overall_attempt < max_overall_retries:
        # Check completion status
        total_locations, successful_lookups, failed_locations = get_completion_status(latlon_to_zip_map)
        
        if failed_locations == 0:
            print(f"\n✅ All {total_locations} locations successfully mapped to zip codes!")
            break
        
        overall_attempt += 1
        print(f"\n" + "=" * 60)
        print(f"=== OVERALL RETRY ATTEMPT {overall_attempt} ===")
        print(f"Status: {successful_lookups}/{total_locations} successful, {failed_locations} still need lookup")
        print("=" * 60)
        
        # Find locations that still need lookup
        failed_pairs = set()
        for (lat, lon), zip_code in latlon_to_zip_map.items():
            if zip_code is None:
                failed_pairs.add((lat, lon))
        
        # Retry failed lookups
        latlon_to_zip_map = retry_missing_zip_lookups(failed_pairs, latlon_to_zip_map, mapping_file, max_retries=5)
        
        # Check final status after this retry attempt
        total_locations, successful_lookups, failed_locations = get_completion_status(latlon_to_zip_map)
        
        if failed_locations == 0:
            print(f"\n✅ All {total_locations} locations successfully mapped after {overall_attempt} overall attempts!")
            break
        elif overall_attempt >= max_overall_retries:
            print(f"\n⚠️  Reached maximum overall retry attempts ({max_overall_retries}).")
            print(f"Final status: {successful_lookups}/{total_locations} successful, {failed_locations} still failed")
            break
        else:
            # Wait longer between overall attempts
            wait_time = min(10 + (overall_attempt * 5), 60)  # Increasing wait time, max 60 seconds
            print(f"\nWaiting {wait_time} seconds before next overall retry attempt...")
            time.sleep(wait_time)
    
    # Add zip columns to datasets
    print(f"\n" + "=" * 60)
    print("=== ADDING ZIP COLUMNS ===")
    print("=" * 60)
    
    files_modified = []
    
    # Process dense dataset
    if not dense_has_zip:
        if add_zip_column(df_dense, 'dense', latlon_to_zip_map):
            print(f"  Writing updated dense dataset...")
            df_dense.to_csv(dense_file, index=False)
            files_modified.append('data_dense.csv')
    elif len(missing_pairs) > 0:
        # Update existing zip column with newly found zip codes
        print(f"  Updating existing zip column in dense dataset...")
        updated_count = 0
        for i, row in df_dense.iterrows():
            if pd.isna(row['zip']) and pd.notna(row['lat']) and pd.notna(row['lon']):
                rounded_lat, rounded_lon = round(row['lat'], 6), round(row['lon'], 6)
                new_zip = latlon_to_zip_map.get((rounded_lat, rounded_lon))
                if new_zip is not None:
                    df_dense.at[i, 'zip'] = new_zip
                    updated_count += 1
        if updated_count > 0:
            print(f"    Updated {updated_count} missing zip codes")
            df_dense.to_csv(dense_file, index=False)
            if 'data_dense.csv' not in files_modified:
                files_modified.append('data_dense.csv')
    
    # Process sparse dataset
    if not sparse_has_zip:
        if add_zip_column(df_sparse, 'sparse', latlon_to_zip_map):
            print(f"  Writing updated sparse dataset...")
            df_sparse.to_csv(sparse_file, index=False)
            files_modified.append('data_sparse.csv')
    elif len(missing_pairs) > 0:
        # Update existing zip column with newly found zip codes
        print(f"  Updating existing zip column in sparse dataset...")
        updated_count = 0
        for i, row in df_sparse.iterrows():
            if pd.isna(row['zip']) and pd.notna(row['lat']) and pd.notna(row['lon']):
                rounded_lat, rounded_lon = round(row['lat'], 6), round(row['lon'], 6)
                new_zip = latlon_to_zip_map.get((rounded_lat, rounded_lon))
                if new_zip is not None:
                    df_sparse.at[i, 'zip'] = new_zip
                    updated_count += 1
        if updated_count > 0:
            print(f"    Updated {updated_count} missing zip codes")
            df_sparse.to_csv(sparse_file, index=False)
            if 'data_sparse.csv' not in files_modified:
                files_modified.append('data_sparse.csv')
    
    # Summary
    print(f"\n" + "=" * 80)
    print("=== PROCESSING COMPLETE ===")
    print("=" * 80)
    
    # Final completion status
    total_locations, successful_lookups, failed_locations = get_completion_status(latlon_to_zip_map)
    
    if files_modified:
        print(f"Files modified: {len(files_modified)}")
        for file in files_modified:
            print(f"  - {file}")
        print(f"\nzip columns have been added/updated in the modified files.")
    else:
        print("No files were modified - zip columns already exist with complete data.")
    
    print(f"\nFinal lookup statistics:")
    print(f"  Total unique locations: {total_locations}")
    print(f"  Successful zip code lookups: {successful_lookups}")
    print(f"  Failed zip code lookups: {failed_locations}")
    
    if failed_locations == 0:
        print(f"  ✅ 100% completion rate achieved!")
    else:
        completion_rate = (successful_lookups / total_locations) * 100 if total_locations > 0 else 0
        print(f"  ⚠️  Completion rate: {completion_rate:.1f}%")

if __name__ == "__main__":
    process_datasets()