from .render import Render
from .model.neuralnet import NeuralNetwork
from .screens.arena import ArenaScreen


def main():
    # zdravko = NeuralNetwork.deserialize('best_so_far/creature.net')
    # ross = NeuralNetwork.deserialize('best_so_far/ross.net')

    with Render(800, 600) as r:
        # r._set_screen(ArenaScreen(ross, zdravko))
        r.loop()


if __name__ == '__main__':
    main()
