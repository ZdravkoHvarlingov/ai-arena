import random
import pickle
import numpy as np
from numpy.core.fromnumeric import size


from .utils import pickle_serialization


class NNRNG:
    def get_random(self, n: int):
        raise NotImplementedError()


class GenericNNRNG(NNRNG):
    def get_random(self, n: int):
        return [random.random() for _ in range(n)]


class NeuralNetwork:
    def __init__(self, creator_tag, activation_func, cross_over, fill_randomly=True):
        self.matrices = []
        self.biases = []
        self.creator_tag = creator_tag
        self.activation_func = activation_func
        self.cross_over_method = cross_over

        # Let's say we have 7 input layers, this should be configurable and not hardcoded
        self.input_nodes = 7
        self.output_nodes = 4
        self.hidden_layers = 2
        self.nodes_per_layer = 14

        if fill_randomly:
            # Every element is a matrix with W(ij) elements
            # W(ij) = the weight from node j to node i
            # In this way if we want to calculate the value of a node i, we can get the ith row ;)
            # We need all the edges going towards a certain node, not all going out of a certain node
            # Bias: last element in the row for the current node.
            
            self.matrices.append(np.random.uniform(low=-1, high=1, size=(self.input_nodes, self.nodes_per_layer)))
            self.biases.append(np.random.uniform(low=-1, high=1, size=self.nodes_per_layer))

            for _ in range(self.hidden_layers - 1):
                self.matrices.append(np.random.uniform(low=-1, high=1, size=(self.nodes_per_layer, self.nodes_per_layer)))
                self.biases.append(np.random.uniform(low=-1, high=1, size=self.nodes_per_layer))
            
            self.matrices.append(np.random.uniform(low=-1, high=1, size=(self.nodes_per_layer, self.output_nodes)))
            self.biases.append(np.random.uniform(low=-1, high=1, size=self.output_nodes))

    def forward(self, input_values):
        '''
            Method performs forward propagation. The input values are supposed to be normalized in the value range [0, 1]
        '''

        current_input = np.array(input_values)
        for idx, matrix in enumerate(self.matrices):
            current_input = np.matmul(current_input, matrix)
            current_input += self.biases[idx]
            current_input = self.activation_func(current_input)

        return current_input

    def cross_over(self, other: "NeuralNetwork"):
        child1 = NeuralNetwork(self.creator_tag, self.activation_func, self.cross_over_method, False)
        child2 = NeuralNetwork(self.creator_tag, self.activation_func, self.cross_over_method, False)

        method = self.cross_over_method()
        for i in range(len(self.matrices)):
            child1.matrices.append(np.zeros(shape=self.matrices[i].shape))
            child1.biases.append(np.zeros(shape=self.biases[i].shape))
            child2.matrices.append(np.zeros(shape=self.matrices[i].shape))
            child2.biases.append(np.zeros(shape=self.biases[i].shape))

            for j in range(self.matrices[i].shape[0]):
                fst, snd = method.perform(self.matrices[i][j], other.matrices[i][j])
                child1.matrices[-1][j] = fst
                child2.matrices[-1][j] = snd
            fst, snd = method.perform(self.biases[i], other.biases[i])
            child1.biases[-1] = fst
            child2.biases[-1] = snd

        return child1, child2

    def serialize(self, filename):
        pickle_serialization(self, filename)

    @staticmethod
    def deserialize(filename):
        with open(filename, 'rb') as binary_file:
            return pickle.load(binary_file)
