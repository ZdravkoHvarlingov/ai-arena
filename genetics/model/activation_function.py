import math


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
def sigmoid(value):
    return 1 / (1 + math.exp(-value))


@FunctionMapper("relu")
def relu(value):
    return value if value > 0 else 0


@FunctionMapper("leaky_relu")
def leaky_relu(value):
    return value if value > 0 else 0.01 * value


@FunctionMapper("tanh")
def tanh(value):
    return 2 / (1 + math.exp(-2 * value)) - 1
