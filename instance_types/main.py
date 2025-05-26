import numpy as np
import pandas as pd

from instance_types.algorithms.euclidean_distance_algorithm import EuclideanDistanceAlgorithm
from instance_types.datasource.csv_dataloader import CSVDataSource

source = CSVDataSource().load("datasource/resources/aws_instance_types.csv")
algorithm = EuclideanDistanceAlgorithm()

data = data2 = {'cpus': [3], 'memory': [7], "gpu_count": [0], "gpu_memory": [0]}

target = pd.DataFrame(data)

instance_type = algorithm.process(source, target)
print(instance_type['name'])
print(instance_type['cpus'])
print(instance_type['memory'])
print(instance_type['gpu_count'])
print(instance_type['gpu_memory'])
