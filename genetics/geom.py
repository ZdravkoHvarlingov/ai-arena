from __future__ import annotations
from typing import List
import math


class Vec2:
    def __init__(self, x: float=0, y: float=0):
        self.x = x
        self.y = y

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def unit(self):
        return self / self.norm()

    def dot(self, v: Vec2) -> float:
        return self.x * v.x + self.y * v.y

    def tolist(self) -> List[float, float]:
        return [self.x, self.y]

    def toilist(self) -> List[int, int]:
        return [int(self.x), int(self.y)]

    def __add__(self, other: Vec2):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Vec2(self.x / other, self.y / other)

    def __str__(self):
        return "Vec2({}, {})".format(self.x, self.y)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        yield self.x
        yield self.y

