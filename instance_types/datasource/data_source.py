import abc
from abc import ABC

import pandas as pd


class DataSource(ABC):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        pass
