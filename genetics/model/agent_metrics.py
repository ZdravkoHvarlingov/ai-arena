class AgentMetrics:
    def __init__(self, agent, frames):
        self.agent = agent
        self.frames = frames
        self._retrieve_metrics()

    def _retrieve_metrics(self):
        self.shots = self.agent.action_counter.get("ShootAction", 0)
        self.moves = self.agent.action_counter.get("MoveAction", 0)
        self.rotations = self.agent.action_counter.get("RotateAction", 0)

        frames = self.frames
        self.moves_time_percent = (self.moves / frames) * 100
        self.rotations_time_percent = (self.rotations / frames) * 100
        self.shots_time_percent = (self.shots / frames) * 100
        self.time_around_a_corner_percent = (self.agent.close_to_corner / frames) * 100
        self.time_around_a_border_percent = (self.agent.close_to_border / frames) * 100
        self.shot_accuracy_percent = (self.agent.successful_shots / self.shots if self.shots > 0 else 0) * 100
        self.shots_during_reload_percent = (self.agent.shots_during_reloading / self.shots if self.shots > 0 else 0) * 100
        self.shots_while_enemy_in_fov_time_percent = (self.agent.shots_while_enemy_in_fov / self.shots if self.shots > 0 else 0) * 100
        self.most_repeated_action_time_percent = (self.agent.most_repeated_counter / frames) * 100
        self.successful_shots = self.agent.successful_shots
        self.bullets_taken = self.agent.bullets_taken
        self.enemy_in_fov_time_percent = (self.agent.is_enemy_in_fov / frames) * 100
        self.enemy_is_close_time_percent = (self.agent.enemy_is_close / frames) * 100
