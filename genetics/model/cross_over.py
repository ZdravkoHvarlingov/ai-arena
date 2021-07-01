import numpy as np


class CrossOverMapper:
    mapper = {}

    def __init__(self, cross_over_type):
        self.cross_over_type = cross_over_type

    def __call__(self, cross_over_class):
        CrossOverMapper.mapper[self.cross_over_type] = cross_over_class

        return cross_over_class

    @classmethod
    def get_cross_over(self, cross_over_type):
        return self.mapper[cross_over_type]


class CrossOver:
    def __init__(self):
        pass

    def perform(self, arr1, arr2):
        raise NotImplementedError()


@CrossOverMapper("two_points")
class TwoPointsCrossOver(CrossOver):
    def __init__(self):
        pass

    def perform(self, arr1, arr2):
        points = np.sort(np.random.choice(len(arr1), 2, replace=False))

        child1 = []
        child2 = []

        child1.extend(arr1[0:points[0] + 1])
        child1.extend(arr2[points[0] + 1:points[1]])
        child1.extend(arr1[points[1]:])

        child2.extend(arr2[0:points[0] + 1])
        child2.extend(arr1[points[0] + 1:points[1]])
        child2.extend(arr2[points[1]:])

        return np.array(child1), np.array(child2)


@CrossOverMapper("arithmetic")
class ArithmeticCrossOver(CrossOver):
    def __init__(self):
        pass

    def perform(self, arr1, arr2):
        a = np.random.uniform(size=1)[0]
        child1 = a * arr1 + (1 - a) * arr2
        child2 = (1 - a) * arr1 + a * arr2

        return child1, child2
