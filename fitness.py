import os

from genetics.model.genetic_evolution import GeneticEvolution
from genetics.model.fitness_func import fitness_func
from genetics.config import Config

from genetics.model.activation_function import FunctionMapper
from genetics.model.selection import SelectionMapper
from genetics.model.mutation import MutationMapper
from genetics.model.cross_over import CrossOverMapper



if __name__ == '__main__':
    genetic_algorithm = GeneticEvolution(
        Config.get("arena.creature_name_tag"),
        Config.get("algorithms.population_size"),
        fitness_func,
        FunctionMapper.get_func(Config.get("algorithms.activation_func")),
        SelectionMapper.get_selection(Config.get("algorithms.selection_type")),
        Config.get("algorithms.previous_generation_rate"),
        MutationMapper.get_mutation(Config.get("algorithms.mutation_type")),
        Config.get("algorithms.mutation_rate"),
        CrossOverMapper.get_cross_over(Config.get("algorithms.cross_over_type")))
    
    gen_serialization_folder = Config.get("serialization.generation_serialization_folder")
    if os.path.exists(os.path.join(gen_serialization_folder, 'generation.data')):
        genetic_algorithm = genetic_algorithm.deserialize(os.path.join(gen_serialization_folder, 'generation.data'))
        print(genetic_algorithm.generations_fitness)
