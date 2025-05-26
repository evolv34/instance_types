import sys

import numpy as np
import pandas as pd

from instance_types.algorithms.algorithm import Algorithm


class EuclideanDistanceAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super(EuclideanDistanceAlgorithm).__init__(**kwargs)

    def process(self, source: pd.DataFrame, target: pd.DataFrame, debug: bool = False):
        x = source[["cpus", "memory", "gpu_count", "gpu_memory"]]
        cpus = x["cpus"] - target["cpus"][0]
        memory = x["memory"] - target["memory"][0]
        gpu_count = x["gpu_count"] - target["gpu_count"][0]
        gpu_memory = x["gpu_memory"] - target["gpu_memory"][0]

        cpus[cpus < 0] = 100000
        memory[memory < 0] = 100000
        gpu_count[gpu_count < 0] = 100000
        gpu_memory[gpu_memory < 0] = 100000

        sq_cpus = cpus ** 2
        sq_memory = memory ** 2
        sq_gpu_count = gpu_count ** 2
        sq_gpu_memory = gpu_memory ** 2

        if debug:
            print(f"sq_cpus {sq_cpus}")
            print("\n\n:::::::::::::::::::::::::::::::::::")
            print(f"sq_memory {sq_memory}")
            print("\n\n:::::::::::::::::::::::::::::::::::")
            print(f"sq_gpu_count {sq_gpu_count}")
            print("\n\n:::::::::::::::::::::::::::::::::::")
            print(f"sq_gpu_memory {sq_gpu_memory}")

        dist_df = np.sqrt(sq_cpus + sq_memory + sq_gpu_count + sq_gpu_count)
        if debug:
            print(f"Total Memory {dist_df}")

        row_num = np.argmin(dist_df)
        if debug:
            print("\n\n:::::::::::::::::::::::::::::::::::")
            print(f"Row Number {row_num}")

        return source.iloc[row_num]
