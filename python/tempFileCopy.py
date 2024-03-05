import os
import shutil

import os
import shutil

def copy_csv_files(path1, path2):
    """
    Copies CSV files from path2 directories to corresponding path1 directories.
    Assumes that each directory in path2 contains a CSV file.
    Does not traverse subdirectories in path2.
    """
    for dir_name in os.listdir(path2):
        dir_path2 = os.path.join(path2, dir_name)
        if os.path.isdir(dir_path2):
            # Check if a CSV file exists in this directory
            csv_files = [f for f in os.listdir(dir_path2) if f.lower().endswith('.csv')]
            if csv_files:
                # Get the corresponding directory name in path1
                target_dir = os.path.join(path1, dir_name)

                # Create the directory in path1 if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)

                # Copy the first CSV file from path2 to path1
                src_file = os.path.join(dir_path2, csv_files[0])
                dst_file = os.path.join(target_dir, csv_files[0])
                shutil.copy2(src_file, dst_file)
                print(f"Copied '{csv_files[0]}' to '{target_dir}'")

# Example usage:
path1 = os.path.join('c:\\','paul', 'sdnist', 'other_techniques')
path2 = os.path.join('c:\\','paul', 'GitHub', 'sdnist-summary', 'results')
copy_csv_files(path1, path2)
