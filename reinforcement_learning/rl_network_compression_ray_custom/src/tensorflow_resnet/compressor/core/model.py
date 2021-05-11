import tensorflow as tf


class Model(object):
    def __init__(self, name, dir, num_layers, params, scope=""):
        self.name = name
        self.num_layers = num_layers
        self.dir = dir
        self.scope = scope
        self.params = params

    def set_name(name):
        self.params["name"] = name
