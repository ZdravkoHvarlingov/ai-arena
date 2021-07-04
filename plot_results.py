import pickle
import matplotlib.pyplot as plt


def load_fitness_history():
    with open('generations/generation_200_700ep_max.data', 'rb') as max_action_file:
        max_action_history = pickle.load(max_action_file).generations_fitness

    with open('generations/generation_200_712ep_probs.data', 'rb') as prob_action_file:
        prob_action_history = pickle.load(prob_action_file).generations_fitness

    return max_action_history[:700], prob_action_history[:700]


if __name__ == '__main__':
    max_action_history, prob_action_history = load_fitness_history()
    plt.plot(max_action_history, label='Action chosen based on max value')
    plt.plot(prob_action_history, label='Action chosen based on probability')

    plt.title('Fitness funtion value')
    plt.ylabel('Fitness value')
    plt.xlabel('Iteration')
    plt.legend()

    plt.savefig('fitness_history.png')
