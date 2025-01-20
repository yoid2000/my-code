import os
import pandas as pd
from pathlib import Path
from syndiffix import Synthesizer

blob_test_env = os.environ.get("BLOB_TEST_PATH")
print(blob_test_env)

blob_test_path = Path(blob_test_env)
data_path = blob_test_path.joinpath('commute', 'commute_table.csv')
df = pd.read_csv(data_path)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

df_syn = Synthesizer(df).sample()