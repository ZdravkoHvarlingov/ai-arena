import random
import pickle
import numpy as np


from .utils import pickle_serialization


class NNRNG:
    def get_random(self, n: int):
        raise NotImplementedError()


class GenericNNRNG(NNRNG):
    def get_random(self, n: int):
        return [random.random() for _ in range(n)]


class NeuralNetwork:
    def __init__(self, creator_tag, hidden_layers, nodes_per_layer, activation_func, cross_over, fill_randomly=True):
        self.weights = []
        self.creator_tag = creator_tag
        self.hidden_layers = hidden_layers
        self.nodes_per_layer = nodes_per_layer
        self.activation_func = activation_func
        self.cross_over_method = cross_over

        # Let's say we have 7 input layers, this should be configurable and not hardcoded
        self.input_nodes = 7
        self.output_nodes = 4

        if fill_randomly:
            # Every element is a matrix with W(ij) elements
            # W(ij) = the weight from node j to node i
            # In this way if we want to calculate the value of a node i, we can get the ith row ;)
            # We need all the edges going towards a certain node, not all going out of a certain node
            # Bias: last element in the row for the current node.

            self.weights.append(np.random.uniform(low=-1, high=1, size=(nodes_per_layer, self.input_nodes + 1)))
            for _ in range(hidden_layers - 1):
                self.weights.append(np.random.uniform(low=-1, high=1, size=(nodes_per_layer, nodes_per_layer + 1)))
            self.weights.append(np.random.uniform(low=-1, high=1, size=(self.output_nodes, nodes_per_layer + 1)))

    def perform_forward_propagation(self, input_values):
        '''
            Method performs forward propagation. The input values are supposed to be normalized in the value range [0, 1]
        '''

        current_input = input_values
        current_input.append(1)
        for layer in self.weights:
            current_input = [self.activation_func(node_weights.dot(current_input)) for node_weights in layer]
            current_input.append(1)

        return current_input[0:-1]

    def cross_over(self, other: "NeuralNetwork"):
        child1 = NeuralNetwork(self.creator_tag, self.hidden_layers, self.nodes_per_layer, self.activation_func, self.cross_over_method, False)
        child2 = NeuralNetwork(self.creator_tag, self.hidden_layers, self.nodes_per_layer, self.activation_func, self.cross_over_method, False)

        method = self.cross_over_method()
        for i in range(len(self.weights)):
            child1.weights.append(np.zeros(shape=self.weights[i].shape))
            child2.weights.append(np.zeros(shape=self.weights[i].shape))
            for j in range(len(self.weights[i])):
                fst, snd = method.perform(self.weights[i][j], other.weights[i][j])
                child1.weights[-1][j] = fst
                child2.weights[-1][j] = snd

        return child1, child2

    def serialize(self, filename):
        pickle_serialization(self, filename)

    @staticmethod
    def deserialize(filename):
        with open(filename, 'rb') as binary_file:
            return pickle.load(binary_file)
