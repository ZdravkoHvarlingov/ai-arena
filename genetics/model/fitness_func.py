from .agent_metrics import AgentMetrics


def fitness_func(nn_to_evaluate_metrics: AgentMetrics, enemy_nn_metrics: AgentMetrics):
    reward = (0.3 * nn_to_evaluate_metrics.shot_accuracy_percent +
              0.1 * nn_to_evaluate_metrics.shots_while_enemy_in_fov_time_percent +
              0.1 * nn_to_evaluate_metrics.enemy_in_fov_time_percent +
              0.5 * (nn_to_evaluate_metrics.moves_time_percent * nn_to_evaluate_metrics.rotations_time_percent))

    if nn_to_evaluate_metrics.shots == 0:
        reward *= 0.2

    if nn_to_evaluate_metrics.time_around_a_border_percent > 30:
        reward *= 0.2

    if nn_to_evaluate_metrics.most_repeated_action_time_percent > 30:
        reward *= 0.1

    if nn_to_evaluate_metrics.enemy_is_close_time_percent > 30:
        reward *= 0.1

    if nn_to_evaluate_metrics.bullets_taken > 0:
        reward /= nn_to_evaluate_metrics.bullets_taken

    return reward
