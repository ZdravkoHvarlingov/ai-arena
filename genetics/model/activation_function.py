import numpy as np


class FunctionMapper:
    mapper = {}

    def __init__(self, activation_type):
        self.activation_type = activation_type

    def __call__(self, func):
        FunctionMapper.mapper[self.activation_type] = func

        return func

    @classmethod
    def get_func(self, activation_type):
        return self.mapper[activation_type]


@FunctionMapper("sigmoid")
def sigmoid(feature_vector):
    return 1 / (1 + np.exp(-feature_vector))


@FunctionMapper("relu")
def relu(feature_vector):
    return np.maximum(feature_vector, 0)


@FunctionMapper("leaky_relu")
def leaky_relu(feature_vector):
    return np.maximum(feature_vector, 0.01 * feature_vector)


@FunctionMapper("tanh")
def tanh(feature_vector):
    return np.tanh(feature_vector)
