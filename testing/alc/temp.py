import os
import pandas as pd
from pathlib import Path


def check_edu_consistency():
    """
    Check that all CSV files under testing/alc/files/attack_files_anon/inputs
    have the same distinct values in the EDU column.
    """
    base_path = Path("files/attack_files_anon/inputs")

    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist")
        return False

    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found")
        return True

    print(f"Found {len(csv_files)} CSV files")

    reference_edu_values = None
    inconsistent_files = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if 'EDU' not in df.columns:
                print(f"Warning: EDU column not found in {csv_file}")
                continue

            current_edu_values = set(df['EDU'].dropna().unique())

            if reference_edu_values is None:
                reference_edu_values = current_edu_values
                print(f"Reference EDU values from {csv_file}: {sorted(reference_edu_values)}")
            else:
                if current_edu_values != reference_edu_values:
                    inconsistent_files.append({
                        'file': csv_file,
                        'values': current_edu_values
                    })

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if inconsistent_files:
        print("\nInconsistencies found:")
        print(f"Reference values: {sorted(reference_edu_values)}")
        for item in inconsistent_files:
            print(f"File: {item['file']}")
            print(f"  Values: {sorted(item['values'])}")
        return False
    else:
        print("\nAll CSV files have consistent EDU values!")
        return True


if __name__ == "__main__":
    check_edu_consistency()


def _select_evenly_distributed_values(sorted_list: list, num_prc_measures: int) -> list[float]:
    '''
    This splits the confidence values into num_prc_measures evenly spaced values.
    Note by the way that this is being used on the attack confidence values.
    '''
    if len(sorted_list) <= num_prc_measures:
        return sorted_list
    selected_values = [sorted_list[0]]
    step_size = (len(sorted_list) - 1) / (num_prc_measures - 1)
    for i in range(1, num_prc_measures - 1):
        index = int(round(i * step_size))
        selected_values.append(sorted_list[index])
    selected_values.append(sorted_list[-1])
    return selected_values


my_list = list(range(1, 100))
# sort my_list
my_list = sorted(my_list, reverse=True)
selected = _select_evenly_distributed_values(my_list, 21)
print(len(selected))