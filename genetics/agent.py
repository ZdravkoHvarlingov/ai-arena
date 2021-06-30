from __future__ import annotations
import numpy as np
import pygame
import math
from typing import List

from .world import CollidableEntity, World, Bullet
from .geom import Vec2
from .model.neuralnet import NeuralNetwork


EPSILON = 0.5
DECEL_FACTOR = 70
MAX_RELOAD_TIME = 1
ANG_DECEL_FACTOR = math.pi * 2
FOV = math.radians(60)
HFOV = FOV / 2.0


class Action:
    def __init__(self):
        pass

    def do(self, agent: Agent):
        raise NotImplementedError()


def to_color(value):
    return 255 - (value * 255), value * 255, 0


def norm_weight(value):
    return (value + 1.0) / 2.0


class Agent(CollidableEntity):
    def __init__(self, world: World, base_pos: Vec2, actions: List[Action], neural_net: NeuralNetwork, color):
        super().__init__(world, base_pos, Vec2(10, 10))
        self.actions = actions
        self.angle = np.random.uniform(low=0, high=math.pi * 2)
        self.linear_speed = 0
        self.angular_speed = 0
        self.reload_timer = 0
        self.neural_net = neural_net
        self.color = color
        self.manual = False

        self.previous_action = -1
        self.bullets_taken = 0
        self.successful_shots = 0
        self.shots_during_reloading = 0
        self.shots_while_enemy_in_fov = 0
        self.close_to_corner = 0
        self.close_to_border = 0
        self.action_counter = {}
        self.same_action_counter = 0
        self.most_repeated_action = -1
        self.most_repeated_counter = 0
        self.is_enemy_in_fov = 0
        self.enemy_is_close = 0

    @property
    def closest_entities(self):
        en_dist = self.world.diagonal
        bul_dist = self.world.diagonal
        enemy = None
        bullet = None

        for e in self.world.entities:
            if isinstance(e, Agent) and e != self:
                edist = (self.position - e.position).norm()
                if edist < en_dist:
                    en_dist = edist
                    enemy = e

            if isinstance(e, Bullet) and e.owner != self:
                bdist = (self.position - e.position).norm()
                if bdist < bul_dist:
                    bul_dist = bdist
                    bullet = e

        return enemy, en_dist, bullet, bul_dist

    @property
    def forward_vector(self):
        return Vec2(math.cos(self.angle), math.sin(self.angle))

    @property
    def nn_inputs(self):
        en, en_dist, bul, bul_dist = self.closest_entities

        dist_enemy = en_dist / self.world.diagonal
        dist_bullet = bul_dist / self.world.diagonal

        # We get weird results when not using unit vectors with atan2
        d = (en.position - self.position).unit()

        ang = self.angle - math.atan2(d.y, d.x)

        # Normalize the angle
        if ang > math.pi or ang < -math.pi:
            ang = math.atan2(math.sin(ang), math.cos(ang))

        # 0.0 = we're 180 degrees away from the target ; 1.0 = spot on
        enemy_in_fov = 1.0 - (abs(ang) / math.pi)

        angle_norm = self.angle / (2.0 * math.pi) % 1

        xnorm = self.position.x / self.world.size.x
        ynorm = self.position.y / self.world.size.y
        reload_norm = self.reload_timer / MAX_RELOAD_TIME

        return [dist_enemy, dist_bullet, xnorm, ynorm, angle_norm, reload_norm, enemy_in_fov]

    @property
    def is_reloading(self):
        return self.reload_timer > 0

    def shoot(self, power: float):
        if self.reload_timer > 0:
            self.shots_during_reloading += 1
            return

        if self.nn_inputs[-1] > 0.8:
            self.shots_while_enemy_in_fov += 1

        self.world.add_entity(Bullet(self.world, self.position + self.forward_vector * 15, self.forward_vector, 250, power, self))
        self.reload_timer = MAX_RELOAD_TIME * power

    def forward(self, speed: float):
        self.linear_speed = speed

    def rotate(self, angular_speed: float):
        self.angular_speed = angular_speed

    def add_action(self, action):
        action_name = type(action).__name__
        if action_name not in self.action_counter:
            self.action_counter[action_name] = 0
        self.action_counter[action_name] += 1

    def adjust_points(self):
        for e in self.world.entities:
            if isinstance(e, Bullet) and e.owner != self and self.check_collisions(e):
                e.mark_destroy = True
                self.bullets_taken += 1
                e.owner.successful_shots += 1

    def check_position(self):
        corner_offset = 0.05 * self.world.diagonal
        x_offset = 0.05 * self.world.size.x
        y_offset = 0.05 * self.world.size.y

        corners = [Vec2(0, 0), Vec2(0, self.world.size.y), Vec2(self.world.size.x, 0), Vec2(self.world.size.x, self.world.size.y)]
        for corner in corners:
            if (self.position - corner).norm() < corner_offset:
                self.close_to_corner += 1

        if (self.position.x < x_offset or
                self.position.x > self.world.size.x - x_offset or
                self.position.y < y_offset or
                self.position.y > self.world.size.y - y_offset):
            self.close_to_border += 1

    def tick(self, delta_s: float):
        previous_position = self.position
        # Check if there's still linear speed to apply
        if self.linear_speed > EPSILON:
            self.position = self.position + (self.forward_vector * self.linear_speed * delta_s)
            self.linear_speed -= DECEL_FACTOR * delta_s
        else:
            self.linear_speed = 0

        if self.position.x + self.half_size.x > self.world.size.x:
            self.position.x = self.world.size.x - self.half_size.x
        elif self.position.x - self.half_size.x < 0:
            self.position.x = self.half_size.x

        if self.position.y + self.half_size.y > self.world.size.y:
            self.position.y = self.world.size.y - self.half_size.y
        elif self.position.y - self.half_size.y < 0:
            self.position.y = self.half_size.y

        nn_inputs = self.nn_inputs
        if nn_inputs[-1] > 0.8:
            self.is_enemy_in_fov += 1
        en, en_dist, _, _ = self.closest_entities
        if self.check_collisions(en):
            self.position = previous_position
        if en_dist < self.half_size.x * 2:
            self.enemy_is_close += 1

        # Check if there's still angular speed to apply
        if abs(self.angular_speed) > EPSILON:
            self.angle += (self.angular_speed * delta_s)

            # Are we going CW or CCW
            if self.angular_speed > 0:
                self.angular_speed -= ANG_DECEL_FACTOR * delta_s
            else:
                self.angular_speed += ANG_DECEL_FACTOR * delta_s
        else:
            self.angular_speed = 0

        if self.reload_timer > 0:
            self.reload_timer -= delta_s

        if self.reload_timer < 0:
            self.reload_timer = 0

        self.adjust_points()
        self.check_position()

        # For manual control only
        if self.manual:
            return

        output = self.neural_net.perform_forward_propagation(nn_inputs)

        best_so_far = -1
        max_value = -1
        for i, o in enumerate(output):
            if max_value < o:
                max_value = o
                best_so_far = i
        self.actions[best_so_far].do(self)
        self.add_action(self.actions[best_so_far])

        if self.previous_action == best_so_far:
            self.same_action_counter += 1
        else:
            self.same_action_counter = 1
        self.previous_action = best_so_far

        if self.same_action_counter > self.most_repeated_counter:
            self.most_repeated_counter = self.same_action_counter
            self.most_repeated_action = best_so_far

    def draw_labels(self, surface):
        font = pygame.font.SysFont('Arial', 20)
        y_spacing = 35

        input_labels = ["     en_dst", "    blt_dst", "             x", "             y", "      angle", "reload_tm", "en_in_fov"]
        for ind, input_label in enumerate(input_labels):
            label = font.render(input_label, True, (0, 0, 0))

            surface.blit(label, (5, 230 + ind * y_spacing))

        output_labels = ["move", "CW", "CCW", "shoot"]
        for ind, output_label in enumerate(output_labels):
            label = font.render(output_label, True, (0, 0, 0))

            surface.blit(label, (264, 283 + ind * y_spacing))

    def draw_nn(self, surface):
        nn_inputs = self.nn_inputs
        y_spacing = 35
        x_spacing = 60
        bx = 50
        by = 50

        graph_h = 15 * max(self.neural_net.nodes_per_layer, self.neural_net.output_nodes, len(nn_inputs))

        for x, layer in enumerate(self.neural_net.weights):
            # sx/sy = current layer middle position
            # psx/psy = previous layer middle position
            sy = by + (graph_h - (len(layer) * y_spacing) / 2.0)
            psy = by + (graph_h - (len(layer[0]) - 1) * y_spacing / 2.0)
            sx = bx + ((x+1) * x_spacing)
            psx = bx + (x * x_spacing)

            for ny, n_node in enumerate(layer):
                for y, weight in enumerate(n_node[:-1]):
                    pygame.draw.line(surface, to_color(norm_weight(weight)), (psx, psy + (y_spacing * y)), (sx, sy + (y_spacing * ny)), 1)

                pygame.draw.circle(surface, (65, 65, 65), (sx, sy + (y_spacing * ny)), 5)

            if x == 0:
                for y in range(len(layer[0]) - 1):
                    pygame.draw.circle(surface, to_color(nn_inputs[y]), (psx, psy + (y_spacing * y)), 5)
            if x == len(self.neural_net.weights) - 1:
                pygame.draw.circle(surface, (0, 255, 0), (sx, sy + (y_spacing * self.previous_action)), 5)

    def draw(self, surface):
        # Draw a solid blue circle in the center
        org = self.position.toilist()
        pygame.draw.circle(surface, self.color, org, 10)

        tip = (self.position + (self.forward_vector * 15)).toilist()
        pygame.draw.line(surface, (0, 0, 0), org, tip, 1)
