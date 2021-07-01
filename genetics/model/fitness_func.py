from .agent_metrics import AgentMetrics


def fitness_func(nn_to_evaluate_metrics: AgentMetrics, enemy_nn_metrics: AgentMetrics):
    bullets_diff = nn_to_evaluate_metrics.successful_shots - nn_to_evaluate_metrics.bullets_taken
    return (10 * bullets_diff + nn_to_evaluate_metrics.moves_time_percent +
            nn_to_evaluate_metrics.rotations_time_percent +
            nn_to_evaluate_metrics.shots_while_enemy_in_fov_time_percent)
