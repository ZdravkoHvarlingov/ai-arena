import os
import pickle
import numpy as np


def get_nearest_combination_k(population_size):
    start = 2
    end = population_size
    res = -1

    while start <= end:
        middle = start + (end - start) // 2  # same as (start + end) / 2 but no overflow is guaranteed

        combinations = middle * (middle - 1)
        if combinations >= population_size:
            res = middle
            end = middle - 1
        else:
            start = middle + 1

    return res


def pickle_serialization(object_to_serialize, filename):
    dirs = filename.split(os.path.sep)[0:-1]
    if len(dirs) > 0:
        dirs = os.path.join(*(filename.split(os.path.sep)[0: -1]))
        os.makedirs(dirs, exist_ok=True)

    with open(filename, 'wb') as binary_file:
        pickle.dump(object_to_serialize, binary_file)
