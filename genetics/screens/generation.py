import pygame
import threading
import logging
import os
import time

from .screen import Screen, Button, Vec2
from genetics.config import Config

from genetics.model.fitness_func import fitness_func
from genetics.model.genetic_evolution import GeneticEvolution
from genetics.model.mutation import MutationMapper
from genetics.model.activation_function import FunctionMapper
from genetics.model.selection import SelectionMapper
from genetics.model.cross_over import CrossOverMapper


logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger()


class GenerationScreen(Screen):
    def __init__(self):
        super().__init__()
        self.font = pygame.font.SysFont('Arial', 40)
        self.render = None

        size = Vec2(100, 60)
        color = (220, 220, 220)
        hover = (150, 150, 150)
        ctext = (0, 0, 0)

        start_btn = Button(Vec2(185, 300), size, color, hover, ctext, "start")
        start_btn.onclick = self.start_training
        pause_btn = Button(Vec2(185 + size.x + 10, 300), size, color, hover, ctext, "pause")
        pause_btn.onclick = self.pause_training
        load_btn = Button(Vec2(185 + 2 * (size.x + 10), 300), size, color, hover, ctext, "load")
        load_btn.onclick = self.load_generation
        fight_btn = Button(Vec2(185 + 3 * (size.x + 10), 300), size, color, hover, ctext, "random fight")
        fight_btn.onclick = self.make_random_fight
        self.register(start_btn)
        self.register(pause_btn)
        self.register(load_btn)
        self.register(fight_btn)

        self._log_debug()
        self.genetic_algorithm = GeneticEvolution(
            Config.get("arena.creature_name_tag"),
            Config.get("algorithms.population_size"),
            fitness_func,
            FunctionMapper.get_func(Config.get("algorithms.activation_func")),
            SelectionMapper.get_selection(Config.get("algorithms.selection_type")),
            Config.get("algorithms.previous_generation_rate"),
            MutationMapper.get_mutation(Config.get("algorithms.mutation_type")),
            Config.get("algorithms.mutation_rate"),
            CrossOverMapper.get_cross_over(Config.get("algorithms.cross_over_type")))

        self.process_n = Config.get("algorithms.training_process_number")
        self.should_train = False
        self.should_join = False
        self.is_joined = True
        self.training_thread = None

    def _log_debug(self):
        logger.debug('Genetic algorithm configuration:')
        logger.debug(f'Population size: {Config.get("algorithms.population_size")}')
        logger.debug(f'Activation function: {Config.get("algorithms.activation_func")}')
        logger.debug(f'Selection type: {Config.get("algorithms.selection_type")}')
        logger.debug(f'Previous generation rate: {Config.get("algorithms.previous_generation_rate")}')
        logger.debug(f'Mutation type: {Config.get("algorithms.mutation_type")}')
        logger.debug(f'Mutation rate: {Config.get("algorithms.mutation_rate")}')
        logger.debug(f'Cross over type: {Config.get("algorithms.cross_over_type")}')
        logger.debug(f'Training process number: {Config.get("algorithms.training_process_number")}')

    def make_random_fight(self):
        if not self.is_joined:
            return

        net_1 = self.genetic_algorithm.best_network_so_far
        net_2 = self.genetic_algorithm.get_random_network()

        self.render.metadata = {'nn1': net_1, 'nn2': net_2}
        self.render.should_switch = True

    def load_generation(self):
        if not self.is_joined:
            return

        deserialization_folder = Config.get("serialization.generation_deserialization_folder")
        self.genetic_algorithm = GeneticEvolution.deserialize(os.path.join(deserialization_folder, 'generation.data'))

    def pause_training(self):
        self.should_train = False

    def start_training(self):
        if not self.is_joined:
            return

        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.deamon = True
        self.should_train = True
        self.should_join = False
        self.is_joined = False

        self.training_thread.start()

    def train(self):
        serialization_frequency = Config.get("serialization.serialization_frequency")
        gen_serialization_folder = Config.get("serialization.generation_serialization_folder")
        creature_serialization_folder = Config.get("serialization.best_creature_serialization_folder")

        while self.should_train:
            logger.info(f'GENERATION #{self.genetic_algorithm.current_generation} is being evaluated...')
            ev_start = time.time()
            self.genetic_algorithm.evaluate_population(self.process_n)
            ev_end = time.time()
            logger.info(f'EVALUATION FINISHED IN {(ev_end - ev_start):.2f}s')
            logger.info(f'TOP FITNESS FOR GENERATION #{self.genetic_algorithm.current_generation} is {self.genetic_algorithm.top_fitness}\n')

            if self.genetic_algorithm.current_generation % serialization_frequency == 0:
                self.genetic_algorithm.serialize(os.path.join(gen_serialization_folder, 'generation.data'))

                best_so_far = self.genetic_algorithm.best_network_so_far
                best_so_far.serialize(os.path.join(creature_serialization_folder, 'creature.net'))
            
            logger.info(f'GENERATION #{self.genetic_algorithm.current_generation + 1} of size {self.genetic_algorithm._population_size} is being created...')
            try:
                self.genetic_algorithm.create_next_generation()
            except Exception as ex:
                logger.exception('Exception: ')
                logger.error(ex)

            logger.info(f'GENERATION #{self.genetic_algorithm.current_generation} created')

        # The last is always evaluated
        logger.info(f'GENERATION #{self.genetic_algorithm.current_generation} is being evaluated...')
        self.genetic_algorithm.evaluate_population(self.process_n)
        logger.info(f'TOP FITNESS FOR GENERATION #{self.genetic_algorithm.current_generation} is {self.genetic_algorithm.top_fitness}\n')

        self.should_join = True

    def draw_center_text(self, text, font_size, y_pos, surface: pygame.Surface):
        font = pygame.font.SysFont('Arial', font_size)
        label_surface = font.render(text, True, (0, 0, 0))
        label_width, _ = label_surface.get_size()

        window_width, _ = surface.get_size()
        surface.blit(label_surface, ((window_width - label_width) // 2, y_pos))

    def draw(self, surface: pygame.Surface):
        if self.should_join:
            self.training_thread.join()
            self.is_joined = True
            self.should_join = False

        surface.fill((255, 255, 255))
        self.draw_center_text('Training process', 40, 25, surface)
        self.draw_center_text(f'Current generation {self.genetic_algorithm.current_generation}', 20, 250, surface)

        if not self.is_joined:
            self.draw_center_text('Training in progress...', 20, 400, surface)

        super().draw(surface)

    def update(self, delta_ms: float):
        super().update(delta_ms)
