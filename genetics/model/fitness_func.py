from .agent_metrics import AgentMetrics


def fitness_func(nn_to_evaluate_metrics: AgentMetrics, enemy_nn_metrics: AgentMetrics):
    bullets_diff = nn_to_evaluate_metrics.successful_shots - nn_to_evaluate_metrics.bullets_taken
    
    result = (10 * bullets_diff + 
            0.25 * nn_to_evaluate_metrics.moves_time_percent +
            0.5 * nn_to_evaluate_metrics.rotations_time_percent +
            0.25 * nn_to_evaluate_metrics.shots_while_enemy_in_fov_time_percent)

    result /= max(1, nn_to_evaluate_metrics.enemy_is_close_time_percent)
    result /= max(1, nn_to_evaluate_metrics.time_around_a_corner_percent)
    result /= max(1, nn_to_evaluate_metrics.most_repeated_action_time_percent)

    return max(result, 0)
