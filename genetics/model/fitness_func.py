from .agent_metrics import AgentMetrics


def fitness_func(nn_to_evaluate_metrics: AgentMetrics, enemy_nn_metrics: AgentMetrics):
    bullets_diff = nn_to_evaluate_metrics.successful_shots - nn_to_evaluate_metrics.bullets_taken
    
    if bullets_diff > 0:
        return 3 + bullets_diff
    if bullets_diff == 0:
        return 1
    return 0
