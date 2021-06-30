import pygame
import math
import time

from .screen import Screen
from genetics.config import Config
from genetics.world import World
from genetics.geom import Vec2
from genetics.model.neuralnet import NeuralNetwork
from genetics.agent import Agent
from genetics.actions import ShootAction, RotateAction,  MoveAction
from genetics.input import input_manager
from .screen import Button


KEY_FORWARD = pygame.K_SPACE
KEY_LOOK_R = pygame.K_RIGHT
KEY_LOOK_L = pygame.K_LEFT
KEY_SHOOT = pygame.K_c
KEY_SLOT1 = pygame.K_1
KEY_SLOT2 = pygame.K_2

FIGHT_TIME = 30
MANUAL_CONTROL = False


class ArenaScreen(Screen):
    def __init__(self, nn1: NeuralNetwork, nn2: NeuralNetwork):
        super().__init__()
        self.render = None
        self.world = World(Vec2(300, 100), Vec2(500, 500))

        self.nn1 = nn1
        self.nn2 = nn2
        self.agent1 = Agent(self.world, Vec2(50, 250), [MoveAction(80), RotateAction(math.pi*0.5), RotateAction(-math.pi*0.5), ShootAction(1.0)], nn1, (91, 108, 207))
        self.agent2 = Agent(self.world, Vec2(450, 250), [MoveAction(80), RotateAction(math.pi*0.5), RotateAction(-math.pi*0.5), ShootAction(1.0)], nn2, (216, 43, 83))
        self.world.add_entity(self.agent1)
        self.world.add_entity(self.agent2)
        self.world.set_main_agent(self.agent1)
        self.button = Button(Vec2(695, 13), Vec2(100, 50), (220, 220, 220), (150, 150, 150), (0, 0, 0), "Go back")
        self.button.onclick = self.get_back_to_main_menu
        self.register(self.button)

        self.fight_time = Config.get("arena.fight_duration_in_s")
        self.start = None
        self.is_started = False
        self.should_stop = False

        if MANUAL_CONTROL:
            self.agent1.manual = True
            self.agent2.manual = True

        if self.agent1.neural_net.creator_tag == self.agent2.neural_net.creator_tag:
            self.agent1_name = "Bestie"
            self.agent2_name = "Random dummy"
        else:
            self.agent1_name = self.agent1.neural_net.creator_tag
            self.agent2_name = self.agent2.neural_net.creator_tag

    def get_back_to_main_menu(self):
        self.render.metadata = {}
        self.render.should_switch = True

    def draw_points(self, surface):
        font = pygame.font.SysFont('Arial', 20)
        agent1_name = font.render(self.agent1_name, True, (91, 108, 207))
        agent1_width, _ = agent1_name.get_size()

        points_label = font.render(f"{self.agent1.successful_shots:02d} : {self.agent2.successful_shots:02d}", True, (0, 0, 0))
        points_width, _ = points_label.get_size()

        agent2_name = font.render(self.agent2_name, True, (216, 43, 83))

        surface.blit(agent1_name, (300, 80))
        surface.blit(points_label, (300 + agent1_width + 10, 80))
        surface.blit(agent2_name, (300 + agent1_width + 10 + points_width + 10, 80))

    def draw_timer(self, surface):
        font = pygame.font.SysFont('Arial', 20)
        time_left = self.fight_time - (time.time() - self.start)

        if time_left < 0.1:
            minutes = 0
            seconds = 0
            self.should_stop = True
        else:
            minutes = int(time_left) // 60
            seconds = int(time_left) % 60

        timer_label = font.render(f'Time left: {minutes:02d}:{seconds:02d}', True, (0, 0, 0))
        timer_width, _ = timer_label.get_size()

        surface.blit(timer_label, (800 - timer_width - 5, 80))

    def draw_winner(self, surface):
        if self.agent1.successful_shots == self.agent2.successful_shots:
            text = 'DRAW'
        elif self.agent1.successful_shots > self.agent2.successful_shots:
            text = f'{self.agent1_name} WON'
        else:
            text = f'{self.agent2_name} WON'

        font = pygame.font.SysFont('Arial', 30)

        winner_label = font.render(text, True, (0, 0, 0))
        winner_width, _ = winner_label.get_size()

        surface.blit(winner_label, (550 - winner_width // 2, 40))

    def draw(self, surface: pygame.Surface):
        if self.is_started is False:
            self.is_started = True
            self.start = time.time()

        surface.fill((255, 255, 255))
        self.draw_points(surface)
        self.draw_timer(surface)

        if self.should_stop:
            self.agent1.manual = True
            self.agent2.manual = True
            self.draw_winner(surface)

        self.world.draw(surface)
        super().draw(surface)

    def update(self, delta_ms: float):
        if MANUAL_CONTROL:
            # Check key inputs
            if input_manager.get_key(KEY_FORWARD):
                self.agent1.forward(80)

            if input_manager.get_key(KEY_SHOOT):
                self.agent1.shoot(1.0)
            #
            # if self.get_key(KEY_RANDOM):
            #     self.agent.neural_net = NeuralNetwork("test", 2, 14, ActivationFunction.SIGMOID)

            if input_manager.get_key(KEY_LOOK_R):
                self.agent1.rotate(math.pi*0.75)
            elif input_manager.get_key(KEY_LOOK_L):
                self.agent1.rotate(-math.pi*0.75)

        self.world.update(delta_ms)
        super().update(delta_ms)
