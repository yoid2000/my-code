import os
import shutil

def copy_csv_files(path1, path2):
    """
    Copies CSV files from path2 directories to corresponding path1 directories.
    Assumes that each directory in path2 contains a CSV file.
    Creates directories in path1 with the same names as those in path2.
    """
    for dirpath, dirnames, filenames in os.walk(path2):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                # Get the corresponding directory name in path1
                relative_dir = os.path.relpath(dirpath, path2)
                target_dir = os.path.join(path1, relative_dir)

                # Create the directory in path1 if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)

                # Copy the CSV file from path2 to path1
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(target_dir, filename)
                shutil.copy2(src_file, dst_file)
                print(f"Copied '{filename}' to '{target_dir}'")

# Example usage:
path1 = os.path.join('c:\\','paul', 'sdnist', 'other_techniques')
path2 = os.path.join('c:\\','paul', 'GitHub', 'sdnist-summary', 'results')
copy_csv_files(path1, path2)
