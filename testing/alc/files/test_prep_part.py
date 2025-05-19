import os
import pandas as pd
import random

for thing in ['anon', 'raw',]:
    input_path = os.path.join(f'attack_files_{thing}', 'inputs')
    df_original = pd.read_csv(os.path.join(input_path, f'{thing}_test_original.csv'))
    synthetic_path = os.path.join(input_path, 'synthetic_files')
    os.makedirs(f'attack_files_{thing}_part', exist_ok=True)
    # write df_original to attack_files_anon_part/inputs/original.csv
    os.makedirs(os.path.join(f'attack_files_{thing}_part', 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(f'attack_files_{thing}_part', 'results'), exist_ok=True)
    os.makedirs(os.path.join(f'attack_files_{thing}_part', 'inputs', 'synthetic_files'), exist_ok=True)
    df_original.to_csv(os.path.join(f'attack_files_{thing}_part', 'inputs', f'{thing}_test_original.csv'), index=False)
    file_columns = [[] for _ in range(5)]
    for column in df_original.columns:
        count = random.randint(1, 2)
        # select three file_columns at random:
        indices = random.sample([0,1,2,3,4], count)
        for index in indices:
            file_columns[index].append(column)
    for i in range(5):
        df_syn = pd.read_csv(os.path.join(synthetic_path, f'syn_{i}.csv'))
        columns = file_columns[i]
        # shuffle the strings in columns
        random.shuffle(columns)
        df_out = df_syn[columns]
        df_out.to_csv(os.path.join(f'attack_files_{thing}_part', 'inputs', 'synthetic_files', f'syn_{i}.csv'), index=False)