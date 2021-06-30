import numpy as np
from .utils import get_nearest_combination_k


class SelectionMapper:
    mapper = {}

    def __init__(self, selection_type):
        self.selection_type = selection_type

    def __call__(self, selection_class):
        SelectionMapper.mapper[self.selection_type] = selection_class

        return selection_class

    @classmethod
    def get_selection(self, selection_type):
        return self.mapper[selection_type]


class Selection:
    def __init__(self, population, keep_parent_rate):
        self.population = population
        self.population_size = len(population)
        self.parent_rate = keep_parent_rate

    def next_generation(self):
        raise NotImplementedError()

    def _choose_parents(self):
        raise NotImplementedError()


@SelectionMapper("pair_best_ones")
class PairBestOnes(Selection):
    def __init__(self, population, keep_parent_rate):
        super().__init__(population, keep_parent_rate)
        self.reproduction_size = get_nearest_combination_k(len(population))

    def next_generation(self):
        '''
            Since the selection receive the population in a sorted way considering the fitness value, the best K neural networks are chosen.
            All possible pairs are generated until the new generation size exceed the required size.
            How is K chosen? Binary search power which you can check inside utils.py file.
        '''
        next_generation = self._choose_parents()
        for i in range(self.reproduction_size):
            for j in range(i + 1, self.reproduction_size):
                child1, child2 = self.population[i][0].cross_over(self.population[j][0])
                next_generation.extend([(child1, 0), (child2, 0)])

        return next_generation[0: self.population_size]

    def _choose_parents(self):
        if self.parent_rate == 0:
            return []

        end = int(len(self.population) * self.parent_rate)
        return self.population[0:end]


@SelectionMapper("roulette")
class RouletteSelection(Selection):
    def __init__(self, population, keep_parent_rate):
        super().__init__(population, keep_parent_rate)
        self._create_wheel()

    def next_generation(self):
        '''
            A neural net is chosen for reproduction according the probability it has.
            The probability is calculated based on the fitness value.
        '''
        next_generation = self._choose_parents()
        while len(next_generation) < self.population_size:
            net1_id = self._choose_element()
            net2_id = self._choose_element()
            if net1_id == net2_id:
                continue

            child1, child2 = self.population[net1_id][0].cross_over(self.population[net2_id][0])
            next_generation.extend([(child1, 0), (child2, 0)])

        return next_generation[0: self.population_size]

    def _create_wheel(self):
        wheel = list(map(lambda pair: pair[1], self.population))
        for i in range(1, len(wheel)):
            wheel[i] += wheel[i - 1]

        self.wheel = wheel

    def _choose_element(self):
        dice = np.random.uniform(low=0, high=self.wheel[-1])
        start = 0
        end = len(self.wheel)

        neural_net = -1
        while start <= end:
            middle = start + (end - start) // 2

            # We want to get the first bigger than the dice
            if self.wheel[middle] >= dice:
                neural_net = middle
                end = middle - 1
            else:
                start = middle + 1

        return neural_net

    def _choose_parents(self):
        if self.parent_rate == 0:
            return []

        amount = int(len(self.population) * self.parent_rate)
        return [self.population[self._choose_element()] for _ in range(amount)]
