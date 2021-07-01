import multiprocessing
import pickle
import logging

import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .evaluation_arena import EvaluationArena
from .neuralnet import NeuralNetwork
from .utils import pickle_serialization

NUMBER_OF_FRAMES = 1800

logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger()


def process_func(pairs, results_queue: multiprocessing.Queue, fitness_func):
    for pair in pairs:
        nn1 = pair[0]
        nn2 = pair[1]

        try:
            arena = EvaluationArena(nn1, nn2)
            nn1_metrics, nn2_metrics = arena.perform_fight(NUMBER_OF_FRAMES)
        except Exception as error:
            logger.exception('Exception')
            print(f'Error happened while simulating the arena: {error}')

        score = fitness_func(nn1_metrics, nn2_metrics)
        results_queue.put((nn1, score))

        score = fitness_func(nn2_metrics, nn1_metrics)
        results_queue.put((nn1, score))

class GeneticEvolution:
    def __init__(self, creator_tag, population_size, fitness_func,
                 activation_func, selection_algorithm, selection_parent_rate, mutation_algorithm, mutation_rate, cross_over):
        self._generation = 1
        self._population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.mutation_rate = mutation_rate
        self.fitness_func = fitness_func
        self.selection_algorithm = selection_algorithm
        self.selection_parent_rate = selection_parent_rate
        self.mutation_algorithm = mutation_algorithm
        self.is_evaluated = False

        self._population = [(NeuralNetwork(creator_tag, activation_func, cross_over), 0) for _ in range(self._population_size)]

    @property
    def current_generation(self):
        return self._generation

    @property
    def best_network_so_far(self):
        return self._population[0][0]

    @property
    def top_fitness(self):
        return self._population[0][1]

    @property
    def mutation(self):
        return self.mutation_rate

    def get_random_network(self):
        net_id = np.random.choice(self._population_size, 1)[0]

        return self._population[net_id][0]

    def evaluate_population(self, number_of_processes):
        nets = list(map(lambda pair: pair[0], self._population))
        np.random.shuffle(nets)
        middle = len(nets) // 2
        pairs = list(zip(nets[:middle], nets[middle:]))

        batch_size = len(pairs) // number_of_processes
        if batch_size * number_of_processes != len(pairs):
            batch_size += 1
        logger.info(f'Evaluating all agents with {batch_size} pairs per process')

        results_queue = multiprocessing.Queue()
        processes = []
        for idx in range(number_of_processes):
            batch_start = idx * batch_size
            batch_end = min((idx + 1) * batch_size, len(pairs))
            process_pairs = pairs[batch_start: batch_end]
            logger.info(f'Process {idx} is taking pairs from {batch_start} to {batch_end - 1} inclusive')

            process = multiprocessing.Process(target=process_func, args=(process_pairs, results_queue, self.fitness_func))
            processes.append(process)
            process.daemon = True
            
            process.start()

        results = []
        for _ in tqdm(range(len(self._population))):
            net_and_score = results_queue.get(block=True, timeout=None)
            results.append(net_and_score)  
        results.sort(key=lambda x: x[1], reverse=True)

        self._population = results
        self.is_evaluated = True

    def create_next_generation(self):
        selection_performer = self.selection_algorithm(self._population, self.selection_parent_rate)
        self._population = selection_performer.next_generation()
        self._perform_mutation()

        self._generation += 1
        self.is_evaluated = False

    def _perform_mutation(self):
        mutation = self.mutation_algorithm(self.mutation_rate)
        for net_pair in self._population:
            mutation.mutate(net_pair[0])

    def serialize(self, filename):
        pickle_serialization(self, filename)

    @staticmethod
    def deserialize(filename):
        with open(filename, 'rb') as binary_file:
            return pickle.load(binary_file)
