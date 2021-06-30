import pygame
from .input import input_manager
from .screens.arena import ArenaScreen
from .screens.generation import GenerationScreen


class Render:
    def __init__(self, width, height):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.cur_screen = None

        self.clock = pygame.time.Clock()

        self.main_screen = GenerationScreen()
        self.should_switch = False
        self.metadata = {}
        self._set_screen(self.main_screen)

    def _set_screen(self, screen):
        screen.render = self
        self.cur_screen = screen

    def switch(self):
        if self.metadata == {}:
            self._set_screen(self.main_screen)
        else:
            self._set_screen(ArenaScreen(**self.metadata))

    def loop(self):
        # Run until the user asks to quit
        running = True
        while running:
            # Limit framerate (And CPU usage :) )
            dt = self.clock.tick(60) / 1000.0

            # Unstack event data
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                    input_manager.handle_event(event)

            if self.should_switch:
                self.switch()
                self.should_switch = False

            if self.cur_screen:
                self.cur_screen.update(dt)

                self.screen.fill((255, 255, 255))
                self.cur_screen.draw(self.screen)

            # # Update entities
            # self.world.update(delta_s=dt*10)
            #
            # # Check key inputs
            # if self.get_key(KEY_FORWARD):
            #     self.agent.forward(80)
            #
            # if self.get_key(KEY_SHOOT) and not self.agent.is_reloading:
            #     self.agent.shoot(1.0)
            # #
            # # if self.get_key(KEY_RANDOM):
            # #     self.agent.neural_net = NeuralNetwork("test", 2, 14, ActivationFunction.SIGMOID)
            #
            # if self.get_key(KEY_LOOK_R):
            #     self.agent.rotate(math.pi*0.75)
            # elif self.get_key(KEY_LOOK_L):
            #     self.agent.rotate(-math.pi*0.75)
            #
            # # Draw the world
            # self.screen.fill((255, 255, 255))
            # self.world.draw(self.screen)

            # Flip the display
            pygame.display.flip()

    def __enter__(self):
        self.screen = pygame.display.set_mode((self.width, self.height))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Done! Time to quit.
        pygame.quit()
