from .agent import Action, Agent


class ShootAction(Action):
    def __init__(self, power: float):
        super().__init__()
        self.power = power
        self.name = 'shoot'

    def do(self, agent: Agent):
        agent.shoot(self.power)


class RotateAction(Action):
    def __init__(self, angular_speed: float):
        super().__init__()
        self.speed = angular_speed
        self.name = f'rt {angular_speed:.2f}'

    def do(self, agent: Agent):
        agent.rotate(self.speed)


class MoveAction(Action):
    def __init__(self, speed: float):
        super().__init__()
        self.speed = speed
        self.name = 'move'

    def do(self, agent: Agent):
        agent.forward(self.speed)
