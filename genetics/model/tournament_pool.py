import pickle
import os


# TO DO
class TournamentPool:
    def __init__(self, directory):
        self.directory = directory

        self.neural_nets = self.load_pool()

    def load_pool(self):
        # We walk only through the root directory
        for (_, _, filenames) in os.walk(self.directory):
            neural_nets = []
            for filename in filenames:
                with open(os.path.join(self.directory, filename), 'rb') as binary_file:
                    neural_nets.append(pickle.load(binary_file))
            return neural_nets
