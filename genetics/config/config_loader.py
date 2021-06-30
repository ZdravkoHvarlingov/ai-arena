import json
import jmespath


class ConfigLoader():
    def __init__(self, path):
        self.path = path
        self.config = {}

    def load(self):
        with open(self.path) as f:
            self.config = json.load(f)

    def get(self, param):
        return jmespath.search(param, self.config)
