import pandas as pd
import os
import numpy as np
from syndiffix import Synthesizer
from syndiffix.stitcher import stitch

def process_dataset(input_csv_path, output_dir, dataset_name):
    """
    Process a single dataset (dense or sparse) to generate synthetic data.
    
    Args:
        input_csv_path: Path to input CSV file
        output_dir: Directory to save parquet files
        dataset_name: Name identifier for the dataset (e.g., 'dense' or 'sparse')
    """
    print(f"\n{'='*60}")
    print(f"=== PROCESSING {dataset_name.upper()} DATASET ===")
    print(f"Input: {input_csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Read the data CSV file
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found: {input_csv_path}")
        return
    
    df = pd.read_csv(input_csv_path)
    
    # Display basic info about the dataset
    print(f"\nDataset info for {dataset_name}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define parquet file paths with dataset name prefix
    df_cell_path = os.path.join(output_dir, f'cell_{dataset_name}.parquet')
    df_cell_sal_path = os.path.join(output_dir, f'cell_sal_{dataset_name}.parquet')
    df_sal_path = os.path.join(output_dir, f'sal_{dataset_name}.parquet')
    df_cell_sal_stitched_path = os.path.join(output_dir, f'cell_sal_stitched_{dataset_name}.parquet')
    df_cell_cnt_path = os.path.join(output_dir, f'cell_cnt_{dataset_name}.parquet')
    
    # Additional paths for zip-based datasets (if zip column exists)
    df_zip_path = os.path.join(output_dir, f'zip_{dataset_name}.parquet')
    df_zip_sal_path = os.path.join(output_dir, f'zip_sal_{dataset_name}.parquet')
    
    print(f"\n--- SYNTHESIZING {dataset_name.upper()} DATAFRAMES ---")
    
    # 1. df_cell - synthesizes column Gitterzelle only with safe-values option
    print(f"\n1. Processing df_cell_{dataset_name} (Gitterzelle only, safe-values)...")
    if os.path.exists(df_cell_path):
        print(f"   Loading existing df_cell_{dataset_name} from: {df_cell_path}")
        df_cell = pd.read_parquet(df_cell_path)
        print(f"   df_cell_{dataset_name} loaded with {len(df_cell)} rows")
    else:
        print(f"   Creating new df_cell_{dataset_name}...")
        df_cell = Synthesizer(df[['Gitterzelle']], value_safe_columns=['Gitterzelle']).sample()
        print(f"   df_cell_{dataset_name} created with {len(df_cell)} rows")
    
    # 2. df_cell_sal - synthesizes Gitterzelle and Gesamtbetrag_Einkuenfte with safe-values for Gitterzelle
    print(f"\n2. Processing df_cell_sal_{dataset_name} (Gitterzelle + Gesamtbetrag_Einkuenfte, safe-values for Gitterzelle)...")
    if os.path.exists(df_cell_sal_path):
        print(f"   Loading existing df_cell_sal_{dataset_name} from: {df_cell_sal_path}")
        df_cell_sal = pd.read_parquet(df_cell_sal_path)
        print(f"   df_cell_sal_{dataset_name} loaded with {len(df_cell_sal)} rows")
    else:
        print(f"   Creating new df_cell_sal_{dataset_name}...")
        df_cell_sal = Synthesizer(df[['Gitterzelle', 'Gesamtbetrag_Einkuenfte']], value_safe_columns=['Gitterzelle']).sample()
        print(f"   df_cell_sal_{dataset_name} created with {len(df_cell_sal)} rows")
    
    # 3. df_sal - synthesizes Gesamtbetrag_Einkuenfte only, no safe-values
    print(f"\n3. Processing df_sal_{dataset_name} (Gesamtbetrag_Einkuenfte only, no safe-values)...")
    if os.path.exists(df_sal_path):
        print(f"   Loading existing df_sal_{dataset_name} from: {df_sal_path}")
        df_sal = pd.read_parquet(df_sal_path)
        print(f"   df_sal_{dataset_name} loaded with {len(df_sal)} rows")
    else:
        print(f"   Creating new df_sal_{dataset_name}...")
        df_sal = Synthesizer(df[['Gesamtbetrag_Einkuenfte']]).sample()
        print(f"   df_sal_{dataset_name} created with {len(df_sal)} rows")
    
    # 4. df_cell_sal_stitched - stitched version
    print(f"\n4. Processing df_cell_sal_stitched_{dataset_name} (stitched from df_cell and df_cell_sal)...")
    if os.path.exists(df_cell_sal_stitched_path):
        print(f"   Loading existing df_cell_sal_stitched_{dataset_name} from: {df_cell_sal_stitched_path}")
        df_cell_sal_stitched = pd.read_parquet(df_cell_sal_stitched_path)
        print(f"   df_cell_sal_stitched_{dataset_name} loaded with {len(df_cell_sal_stitched)} rows")
    else:
        print(f"   Creating new df_cell_sal_stitched_{dataset_name}...")
        df_cell_sal_stitched = stitch(df_left=df_cell, df_right=df_cell_sal, shared=False)
        print(f"   df_cell_sal_stitched_{dataset_name} created with {len(df_cell_sal_stitched)} rows")
    
    # 5. df_cell_cnt - no safe values
    df_cell_cnt_syn = None
    print(f"\n5. Processing df_cell_cnt_{dataset_name} (cell_count only, no safe-values)...")
    if os.path.exists(df_cell_cnt_path):
        print(f"   Loading existing df_cell_cnt_{dataset_name} from: {df_cell_cnt_path}")
        df_cell_cnt = pd.read_parquet(df_cell_cnt_path)
        print(f"   df_cell_cnt_{dataset_name} loaded with {len(df_cell_cnt)} rows")
    else:
        # Make sure column 'cell_count' exists in df
        if 'cell_count' not in df.columns:
            print(f"    cell_count column not found in {dataset_name} dataset, skipping")
        else:
            print(f"   Creating new df_cell_cnt_{dataset_name}...")
            df_cell_cnt_syn = Synthesizer(df[['cell_count']]).sample()
            print(f"   df_cell_cnt_syn_{dataset_name} created with {len(df_cell_cnt_syn)} rows")
    
    # 6. df_zip - zip column only (if zip column exists)
    df_zip = None
    print(f"\n6. Processing df_zip_{dataset_name} (zip only, no safe-values)...")
    if 'zip' not in df.columns:
        print(f"    zip column not found in {dataset_name} dataset, skipping zip-based syntheses")
    elif os.path.exists(df_zip_path):
        print(f"   Loading existing df_zip_{dataset_name} from: {df_zip_path}")
        df_zip = pd.read_parquet(df_zip_path)
        print(f"   df_zip_{dataset_name} loaded with {len(df_zip)} rows")
    else:
        print(f"   Creating new df_zip_{dataset_name}...")
        df_zip = Synthesizer(df[['zip']]).sample()
        print(f"   df_zip_{dataset_name} created with {len(df_zip)} rows")
    
    # 7. df_zip_sal - zip and Gesamtbetrag_Einkuenfte columns (if zip column exists)
    df_zip_sal = None
    print(f"\n7. Processing df_zip_sal_{dataset_name} (zip + Gesamtbetrag_Einkuenfte, no safe-values)...")
    if 'zip' not in df.columns:
        print(f"    zip column not found in {dataset_name} dataset, skipping")
    elif os.path.exists(df_zip_sal_path):
        print(f"   Loading existing df_zip_sal_{dataset_name} from: {df_zip_sal_path}")
        df_zip_sal = pd.read_parquet(df_zip_sal_path)
        print(f"   df_zip_sal_{dataset_name} loaded with {len(df_zip_sal)} rows")
    else:
        print(f"   Creating new df_zip_sal_{dataset_name}...")
        df_zip_sal = Synthesizer(df[['zip', 'Gesamtbetrag_Einkuenfte']]).sample()
        print(f"   df_zip_sal_{dataset_name} created with {len(df_zip_sal)} rows")
    
    print(f"\n--- SAVING {dataset_name.upper()} DATAFRAMES ---")
    print(f"\nSaving synthesized dataframes to parquet files in {output_dir}...")
    
    # Save df_cell_cnt_syn (only if it was newly created)
    if not os.path.exists(df_cell_cnt_path):
        if df_cell_cnt_syn is not None:
            df_cell_cnt_syn.to_parquet(df_cell_cnt_path, index=False)
            print(f"Saved df_cell_cnt_{dataset_name} to: {df_cell_cnt_path}")
        else:
            print(f"    cell_count column not found in {dataset_name} dataset, skipping")
    else:
        print(f"df_cell_cnt_{dataset_name} already exists at: {df_cell_cnt_path} (skipping save)")
    
    # Save df_cell (only if it was newly created)
    if not os.path.exists(df_cell_path):
        df_cell.to_parquet(df_cell_path, index=False)
        print(f"Saved df_cell_{dataset_name} to: {df_cell_path}")
    else:
        print(f"df_cell_{dataset_name} already exists at: {df_cell_path} (skipping save)")
    
    # Save df_cell_sal (only if it was newly created)
    if not os.path.exists(df_cell_sal_path):
        df_cell_sal.to_parquet(df_cell_sal_path, index=False)
        print(f"Saved df_cell_sal_{dataset_name} to: {df_cell_sal_path}")
    else:
        print(f"df_cell_sal_{dataset_name} already exists at: {df_cell_sal_path} (skipping save)")
    
    # Save df_sal (only if it was newly created)
    if not os.path.exists(df_sal_path):
        df_sal.to_parquet(df_sal_path, index=False)
        print(f"Saved df_sal_{dataset_name} to: {df_sal_path}")
    else:
        print(f"df_sal_{dataset_name} already exists at: {df_sal_path} (skipping save)")
    
    # Save df_cell_sal_stitched (only if it was newly created)
    if not os.path.exists(df_cell_sal_stitched_path):
        df_cell_sal_stitched.to_parquet(df_cell_sal_stitched_path, index=False)
        print(f"Saved df_cell_sal_stitched_{dataset_name} to: {df_cell_sal_stitched_path}")
    else:
        print(f"df_cell_sal_stitched_{dataset_name} already exists at: {df_cell_sal_stitched_path} (skipping save)")
    
    # Save df_zip (only if it was newly created and zip column exists)
    if 'zip' in df.columns:
        if not os.path.exists(df_zip_path):
            if df_zip is not None:
                df_zip.to_parquet(df_zip_path, index=False)
                print(f"Saved df_zip_{dataset_name} to: {df_zip_path}")
        else:
            print(f"df_zip_{dataset_name} already exists at: {df_zip_path} (skipping save)")
    
    # Save df_zip_sal (only if it was newly created and zip column exists)
    if 'zip' in df.columns:
        if not os.path.exists(df_zip_sal_path):
            if df_zip_sal is not None:
                df_zip_sal.to_parquet(df_zip_sal_path, index=False)
                print(f"Saved df_zip_sal_{dataset_name} to: {df_zip_sal_path}")
        else:
            print(f"df_zip_sal_{dataset_name} already exists at: {df_zip_sal_path} (skipping save)")
    
    print(f"\n{dataset_name.upper()} dataset processing completed successfully!")
    
    # Return summary statistics
    summary = {
        'dataset_name': dataset_name,
        'original_shape': df.shape,
        'has_zip_column': 'zip' in df.columns,
        'df_cell_shape': df_cell.shape if 'df_cell' in locals() else None,
        'df_cell_sal_shape': df_cell_sal.shape if 'df_cell_sal' in locals() else None,
        'df_sal_shape': df_sal.shape if 'df_sal' in locals() else None,
        'df_cell_sal_stitched_shape': df_cell_sal_stitched.shape if 'df_cell_sal_stitched' in locals() else None,
        'df_zip_shape': df_zip.shape if df_zip is not None else None,
        'df_zip_sal_shape': df_zip_sal.shape if df_zip_sal is not None else None,
        'original_unique_gitterzelle': df['Gitterzelle'].nunique(),
        'df_cell_unique_gitterzelle': df_cell['Gitterzelle'].nunique() if 'df_cell' in locals() else None,
        'df_cell_sal_unique_gitterzelle': df_cell_sal['Gitterzelle'].nunique() if 'df_cell_sal' in locals() else None,
        'df_cell_sal_stitched_unique_gitterzelle': df_cell_sal_stitched['Gitterzelle'].nunique() if 'df_cell_sal_stitched' in locals() else None,
        'original_unique_zip': df['zip'].nunique() if 'zip' in df.columns else None,
        'df_zip_unique_zip': df_zip['zip'].nunique() if df_zip is not None else None,
        'df_zip_sal_unique_zip': df_zip_sal['zip'].nunique() if df_zip_sal is not None else None
    }
    
    return summary

def main():
    """Main function to process both dense and sparse datasets."""
    print("="*80)
    print("=== DUAL DATASET SYNTHETIC DATA GENERATION ===")
    print("="*80)
    
    # Define input and output paths
    dense_input = os.path.join('salary_data2', 'data_dense.csv')
    sparse_input = os.path.join('salary_data2', 'data_sparse.csv')
    output_dir = 'salary_data2'
    
    # Check if input files exist
    if not os.path.exists(dense_input):
        print(f"Error: Dense dataset not found: {dense_input}")
        return
    
    if not os.path.exists(sparse_input):
        print(f"Error: Sparse dataset not found: {sparse_input}")
        return
    
    print(f"Input files:")
    print(f"  Dense:  {dense_input}")
    print(f"  Sparse: {sparse_input}")
    print(f"Output directory: {output_dir}")
    
    # Process both datasets
    summaries = []
    
    # Process dense dataset
    dense_summary = process_dataset(dense_input, output_dir, 'dense')
    summaries.append(dense_summary)
    
    # Process sparse dataset
    sparse_summary = process_dataset(sparse_input, output_dir, 'sparse')
    summaries.append(sparse_summary)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("=== FINAL SUMMARY ===")
    print(f"{'='*80}")
    
    for summary in summaries:
        dataset_name = summary['dataset_name']
        has_zip = summary['has_zip_column']
        
        print(f"\n{dataset_name.upper()} DATASET:")
        print(f"  Original data:           {summary['original_shape'][0]:,} rows × {summary['original_shape'][1]} columns")
        print(f"  Has zip column:          {'Yes' if has_zip else 'No'}")
        
        if summary['df_cell_shape']:
            print(f"  df_cell_{dataset_name}:             {summary['df_cell_shape'][0]:,} rows × {summary['df_cell_shape'][1]} columns")
        if summary['df_cell_sal_shape']:
            print(f"  df_cell_sal_{dataset_name}:         {summary['df_cell_sal_shape'][0]:,} rows × {summary['df_cell_sal_shape'][1]} columns")
        if summary['df_sal_shape']:
            print(f"  df_sal_{dataset_name}:              {summary['df_sal_shape'][0]:,} rows × {summary['df_sal_shape'][1]} columns")
        if summary['df_cell_sal_stitched_shape']:
            print(f"  df_cell_sal_stitched_{dataset_name}: {summary['df_cell_sal_stitched_shape'][0]:,} rows × {summary['df_cell_sal_stitched_shape'][1]} columns")
        
        # Add zip-based datasets to summary if they exist
        if has_zip:
            if summary['df_zip_shape']:
                print(f"  df_zip_{dataset_name}:              {summary['df_zip_shape'][0]:,} rows × {summary['df_zip_shape'][1]} columns")
            if summary['df_zip_sal_shape']:
                print(f"  df_zip_sal_{dataset_name}:          {summary['df_zip_sal_shape'][0]:,} rows × {summary['df_zip_sal_shape'][1]} columns")
        
        print(f"  Unique Gitterzelle values:")
        print(f"    Original:                {summary['original_unique_gitterzelle']:,}")
        if summary['df_cell_unique_gitterzelle']:
            print(f"    df_cell_{dataset_name}:             {summary['df_cell_unique_gitterzelle']:,}")
        if summary['df_cell_sal_unique_gitterzelle']:
            print(f"    df_cell_sal_{dataset_name}:         {summary['df_cell_sal_unique_gitterzelle']:,}")
        if summary['df_cell_sal_stitched_unique_gitterzelle']:
            print(f"    df_cell_sal_stitched_{dataset_name}: {summary['df_cell_sal_stitched_unique_gitterzelle']:,}")
        
        # Add zip unique values if applicable
        if has_zip:
            print(f"  Unique zip values:")
            print(f"    Original:                {summary['original_unique_zip']:,}")
            if summary['df_zip_unique_zip']:
                print(f"    df_zip_{dataset_name}:              {summary['df_zip_unique_zip']:,}")
            if summary['df_zip_sal_unique_zip']:
                print(f"    df_zip_sal_{dataset_name}:          {summary['df_zip_sal_unique_zip']:,}")
    
    # List generated files
    print(f"\n{'='*80}")
    print("=== GENERATED FILES ===")
    print(f"{'='*80}")
    
    if os.path.exists(output_dir):
        parquet_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
        parquet_files.sort()
        
        dense_files = [f for f in parquet_files if 'dense' in f]
        sparse_files = [f for f in parquet_files if 'sparse' in f]
        
        print(f"\nDENSE dataset files ({len(dense_files)}):")
        for i, f in enumerate(dense_files, 1):
            file_path = os.path.join(output_dir, f)
            file_size = os.path.getsize(file_path) / 1024  # KB
            file_type = ""
            if 'zip_sal_' in f:
                file_type = " [ZIP+SALARY]"
            elif 'zip_' in f:
                file_type = " [ZIP ONLY]"
            elif 'cell_sal_stitched_' in f:
                file_type = " [CELL+SALARY STITCHED]"
            elif 'cell_sal_' in f:
                file_type = " [CELL+SALARY]"
            elif 'cell_cnt_' in f:
                file_type = " [CELL COUNT]"
            elif 'cell_' in f:
                file_type = " [CELL ONLY]"
            elif 'sal_' in f:
                file_type = " [SALARY ONLY]"
            print(f"  {i}. {f} ({file_size:.1f} KB){file_type}")
        
        print(f"\nSPARSE dataset files ({len(sparse_files)}):")
        for i, f in enumerate(sparse_files, 1):
            file_path = os.path.join(output_dir, f)
            file_size = os.path.getsize(file_path) / 1024  # KB
            file_type = ""
            if 'zip_sal_' in f:
                file_type = " [ZIP+SALARY]"
            elif 'zip_' in f:
                file_type = " [ZIP ONLY]"
            elif 'cell_sal_stitched_' in f:
                file_type = " [CELL+SALARY STITCHED]"
            elif 'cell_sal_' in f:
                file_type = " [CELL+SALARY]"
            elif 'cell_cnt_' in f:
                file_type = " [CELL COUNT]"
            elif 'cell_' in f:
                file_type = " [CELL ONLY]"
            elif 'sal_' in f:
                file_type = " [SALARY ONLY]"
            print(f"  {i}. {f} ({file_size:.1f} KB){file_type}")
        
        print(f"\nTotal parquet files generated: {len(parquet_files)}")
    
    print(f"\n{'='*80}")
    print("=== PROCESSING COMPLETE ===")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
