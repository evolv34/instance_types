import numpy as np
import pandas as pd

from instance_types.algorithms.euclidean_distance_algorithm import EuclideanDistanceAlgorithm
from instance_types.datasource.csv_dataloader import CSVDataSource

source = CSVDataSource().load("datasource/resources/aws_instance_types.csv")
algorithm = EuclideanDistanceAlgorithm()

data = data2 = {'cpus': [16], 'memory': [100]}

target = pd.DataFrame(data)

instance_type = algorithm.process(source, target)
print(instance_type)
