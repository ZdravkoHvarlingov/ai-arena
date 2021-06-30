import math
from genetics.world import World
from genetics.geom import Vec2
from genetics.model.neuralnet import NeuralNetwork
from genetics.agent import Agent
from genetics.actions import ShootAction, RotateAction,  MoveAction


from .agent_metrics import AgentMetrics


class EvaluationArena:
    def __init__(self, nn1: NeuralNetwork, nn2: NeuralNetwork):
        super().__init__()
        self.world = World(Vec2(300, 100), Vec2(500, 500), False)

        self.nn1 = nn1
        self.nn2 = nn2
        self.agent1 = Agent(self.world, Vec2(50, 250), [MoveAction(80), RotateAction(math.pi*0.5), RotateAction(-math.pi*0.5), ShootAction(1.0)], nn1, (91, 108, 207))
        self.agent2 = Agent(self.world, Vec2(450, 250), [MoveAction(80), RotateAction(math.pi*0.5), RotateAction(-math.pi*0.5), ShootAction(1.0)], nn2, (216, 43, 83))
        self.world.add_entity(self.agent1)
        self.world.add_entity(self.agent2)
        self.world.set_main_agent(self.agent1)

    def update(self, delta_ms: float):
        self.world.update(delta_ms)

    def perform_fight(self, frames):
        dt_60_fps = 0.016666666666667 * 6

        tick = 0
        while tick < frames:
            self.update(dt_60_fps)
            tick += 1

        agent1_metrics = AgentMetrics(self.agent1, frames)
        agent2_metrics = AgentMetrics(self.agent2, frames)

        return agent1_metrics, agent2_metrics
