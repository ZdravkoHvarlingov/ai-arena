import pygame


class InputManager:
    def __init__(self):
        self.keys = {}

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            self.keys[event.key] = True
        elif event.type == pygame.KEYUP:
            self.keys[event.key] = False

    def get_key(self, key):
        return self.keys.get(key, False)


input_manager = InputManager()
