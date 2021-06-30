import pygame
from genetics.geom import Vec2


class Element:
    def draw(self, surface: pygame.Surface):
        raise NotImplementedError()

    def update(self, delta_ms: float):
        raise NotImplementedError()


class Button(Element):
    def __init__(self, position: Vec2, size: Vec2, color, hover_color, text_color, text):
        self.position = position
        self.size = size
        self.hover_color = hover_color
        self.color = color
        self.text_color = text_color
        self.text = text
        self.font = pygame.font.SysFont("Arial", 20)

        self._rendered = None
        self.is_mouse_over = False

        self._update_text()
        self.onclick = None
        self.clicked = False

    def _update_text(self):
        self._rendered = self.font.render(self.text, True, self.text_color)

    def draw(self, surface: pygame.Surface):
        w, h = self._rendered.get_size()
        bx = self.position.x + ((self.size.x - w) / 2)
        by = self.position.y + ((self.size.y - h) / 2)

        if self.is_mouse_over:
            pygame.draw.rect(surface, self.hover_color, (self.position.x, self.position.y, self.size.x, self.size.y))
        else:
            pygame.draw.rect(surface, self.color, (self.position.x, self.position.y, self.size.x, self.size.y))

        surface.blit(self._rendered, (bx, by))

    def update(self, delta_ms: float):
        mx, my = pygame.mouse.get_pos()
        self.is_mouse_over = (self.position.x <= mx <= (self.position.x + self.size.x)) and (self.position.y <= my <= (self.position.y + self.size.y))

        b0, _, _ = pygame.mouse.get_pressed()
        if self.is_mouse_over and b0 and self.onclick and not self.clicked:
            self.clicked = True
            self.onclick()

        if not b0 and self.clicked:
            self.clicked = False


class Screen(Element):
    def __init__(self):
        self._elements = []

    def register(self, e: Element):
        self._elements.append(e)

    def draw(self, surface: pygame.Surface):
        for e in self._elements:
            e.draw(surface)

    def update(self, delta_ms: float):
        for e in self._elements:
            e.update(delta_ms)
