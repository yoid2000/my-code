import pandas as pd
import random

class TableMaker:
    def __init__(self):
        self.df = None

    def simple_uniform(self, N, C, V):
        """
        Generates a dataframe with N rows, C columns, and V distinct values per column.
        Each value is randomly selected with uniform probability from the set of distinct values.
        Column names are 'C1', 'C2', ..., 'Cn'.
        Args:
            N (int): Number of rows.
            C (int): Number of columns.
            V (int): Number of distinct values per column.
        Returns:
            pd.DataFrame: A dataframe with the specified properties.
        """
        # Generate random values for each cell
        data = [[random.randint(1, V) for _ in range(C)] for _ in range(N)]
        
        # Create column names
        columns = [f"C{i+1}" for i in range(C)]
        
        # Create dataframe
        self.df = pd.DataFrame(data, columns=columns)
        return self.df

# Example usage


if __name__ == "__main__":
    table_maker = TableMaker()
    result_df = table_maker.simple_uniform(N=10, C=5, V=4)
    print(result_df)