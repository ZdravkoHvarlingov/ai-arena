from __future__ import annotations
from .geom import Vec2

import pygame


class World:
    def __init__(self, pos: Vec2, size: Vec2, should_render=True):
        self.entities = []
        self.pos = pos
        self.size = size
        self.diagonal = size.norm()
        self.agent = None

        if should_render:
            self.font = pygame.font.SysFont('Arial', 40)
            self.nn_text = self.font.render('Neural Network', True, (0, 0, 0))
            self.world_text = self.font.render('Arena', True, (0, 0, 0))

    def set_main_agent(self, ent: Entity):
        self.agent = ent

    def add_entity(self, ent: Entity):
        self.entities.append(ent)

    def update(self, delta_s: float):
        for e in self.entities:
            # We destroy entities that were marked as destroyed during the previous update
            if e.mark_destroy:
                self.entities.remove(e)
                continue

            e.tick(delta_s)

    def draw(self, surface):
        black = (0, 0, 0)

        # Draw text
        surface.blit(self.nn_text, (25, 25))
        surface.blit(self.world_text, (300, 25))

        # Draw the Neural Network
        nn_surface = pygame.Surface((300, 500))
        nn_surface.fill((255, 255, 255))
        if self.agent:
            self.agent.draw_nn(nn_surface)
            surface.blit(nn_surface, (25, 100))

        # Draw the world
        world_surface = pygame.Surface(self.size.toilist())
        world_surface.fill((255, 255, 255))
        pygame.draw.line(world_surface, black, (0, 0), (self.size.x-1, 0), 1)
        pygame.draw.line(world_surface, black, (0, self.size.y-1), (self.size.x-1, self.size.y-1), 1)
        pygame.draw.line(world_surface, black, (self.size.x-1, 0), (self.size.x-1, self.size.y-1), 1)
        pygame.draw.line(world_surface, black, (0, 0), (0, self.size.y-1), 1)
        for e in self.entities:
            e.draw(world_surface)

        # Render the world on the screen
        surface.blit(world_surface, self.pos.toilist())


class Entity:
    def __init__(self, world: World, position: Vec2):
        self.world = world
        self.position = position
        self.mark_destroy = False

    def destroy(self):
        self.mark_destroy = True

    def tick(self, delta_s: float):
        return

    def draw(self, surface):
        return


class CollidableEntity(Entity):
    def __init__(self, world: World, position: Vec2, half_size: Vec2):
        super().__init__(world, position)
        self.half_size = half_size

    def check_collisions(self, c2: CollidableEntity):
        distance = (self.position - c2.position).norm()

        # Since the objects are round
        return distance < self.half_size.x + c2.half_size.x


class Bullet(CollidableEntity):
    def __init__(self, world: World, base_pos: Vec2, forward_vector: Vec2, speed: float, power: float, owner: "Agent"):
        super().__init__(world, base_pos, Vec2(2, 2))
        self.forward_vector = forward_vector
        self.speed = speed
        self.power = power
        self.owner = owner

    def tick(self, delta_s: float):
        self.position = self.position + (self.forward_vector * self.speed * delta_s)

        # Out of bounds check
        if self.position.x > self.world.size.x or self.position.x < 0:
            self.destroy()

        if self.position.y > self.world.size.y or self.position.y < 0:
            self.destroy()

    def draw(self, surface):
        pygame.draw.circle(surface, (0, 0, 0), self.position.toilist(), 2)
