from .agent import Action, Agent


class ShootAction(Action):
    def __init__(self, power: float):
        super().__init__()
        self.power = power

    def do(self, agent: Agent):
        agent.shoot(self.power)


class RotateAction(Action):
    def __init__(self, angular_speed: float):
        super().__init__()
        self.speed = angular_speed

    def do(self, agent: Agent):
        agent.rotate(self.speed)


class MoveAction(Action):
    def __init__(self, speed: float):
        super().__init__()
        self.speed = speed

    def do(self, agent: Agent):
        agent.forward(self.speed)
