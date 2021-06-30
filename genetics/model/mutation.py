import numpy as np


class MutationMapper:
    mapper = {}

    def __init__(self, mutation_type):
        self.mutation_type = mutation_type

    def __call__(self, mutation_class):
        MutationMapper.mapper[self.mutation_type] = mutation_class

        return mutation_class

    @classmethod
    def get_mutation(self, mutation_type):
        return self.mapper[mutation_type]


class Mutation:
    def __init__(self, rate):
        self.rate = rate

    def mutate(self, neural_net):
        raise NotImplementedError()


@MutationMapper("single_weight_per_node")
class SingleWeightPerNodeMutation(Mutation):
    def __init__(self, rate):
        super().__init__(rate)

    def mutate(self, neural_net):
        '''
            Given a mutation rate, the mutation goes through each neural net.
            If the neural network is chosen for mutation one weight per node is regenerated.
        '''
        if np.random.uniform(low=0, high=1) <= self.rate:
            for layer in neural_net.weights:
                for node_weights in layer:
                    weight_to_mutate = np.random.choice(len(node_weights), 1)[0]
                    node_weights[weight_to_mutate] = np.random.uniform(low=-1, high=1)


@MutationMapper("all_weights_mutation")
class AllWeightsMutation(Mutation):
    def __init__(self, rate):
        super().__init__(rate)

    def mutate(self, neural_net):
        '''
            Given a mutation rate, the mutation goes through each weight inside the neural net.
            Depending on the mutation rate some of them are regenerated.
        '''
        for layer in neural_net.weights:
            for node_weights in layer:
                for i, _ in enumerate(node_weights):
                    if np.random.uniform(low=0, high=1) <= self.rate:
                        node_weights[i] = np.random.uniform(low=-1, high=1)
