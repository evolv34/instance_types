import sys

import numpy as np
import pandas as pd

from instance_types.algorithms.algorithm import Algorithm


class EuclideanDistanceAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super(EuclideanDistanceAlgorithm).__init__(**kwargs)

    def process(self, source: pd.DataFrame, target: pd.DataFrame, debug: bool = False) -> str:
        x = source[["cpus", "memory"]]
        cpus = x["cpus"] - target["cpus"][0]
        memory = x["memory"] - target["memory"][0]

        cpus[cpus < 0] = 100000
        memory[memory < 0] = 100000

        sq_cpus = cpus ** 2
        sq_memory = memory ** 2

        if debug:
            print(f"sq_cpus {sq_cpus}")
            print("\n\n:::::::::::::::::::::::::::::::::::")
            print(f"sq_memory {sq_memory}")

        dist_df = np.sqrt(sq_cpus + sq_memory)
        if debug:
            print(f"Total Memory {dist_df}")

        row_num = np.argmin(dist_df)
        if debug:
            print("\n\n:::::::::::::::::::::::::::::::::::")
            print(f"Row Number {row_num}")

        return source["name"][row_num]
