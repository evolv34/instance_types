import pandas as pd

from instance_types.datasource.data_source import DataSource


class CSVDataSource(DataSource):

    def __init__(self):
        super().__init__()

    def load(self, path) -> pd.DataFrame:
        return pd.read_csv(path)
