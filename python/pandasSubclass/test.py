import pandas as pd
import numpy as np
import pandas_sdx

# Create a DataFrame
df = pd.DataFrame({'col1': np.arange(5), 'col2': np.arange(5), 'col3': np.arange(5), 'col4': np.arange(5)})

# Convert the DataFrame to MyPD
df_sdx = pandas_sdx.PandasSdx(df)

# Test the new behavior
df1 = df_sdx[['col1', 'col2']]
print(f"type of df1 is {type(df1)}")
print(df1)
df2 = df_sdx.iloc[:, [1, 3]]
print(f"type of df2 is {type(df2)}")
print(df2)

# Test the new behavior
df3 = df[['col1', 'col2']]
print(f"type of df3 is {type(df3)}")
print(df3)
df4 = df.iloc[:, [1, 3]]
print(f"type of df4 is {type(df4)}")
print(df4)