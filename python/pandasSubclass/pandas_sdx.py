import pandas as pd

class PandasSdx:
    def __init__(self, df:pd.DataFrame) -> None:
        self.df = df

    def __getitem__(self, item):
        print(item)

    class iloc:
        @staticmethod
        def __getitem__(item):
            print(item)
