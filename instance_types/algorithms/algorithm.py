import abc
from abc import ABC


class Algorithm(ABC):

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def process(self, **kwargs) -> str:
        pass
