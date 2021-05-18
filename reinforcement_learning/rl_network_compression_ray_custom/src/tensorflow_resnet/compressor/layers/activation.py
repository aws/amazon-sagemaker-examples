import tensorflow as tf

from ..core import Fake, Layer


class ReLU(Layer):
    """Definition for a ReLU layer.

    Args:
        name: name of the module.
        inputs: the input symbol.
        start: layer number for begining.
        end: layer number for ending.
    """

    def __init__(self, name, inputs, start=None, end=None, fake=False):
        super(ReLU, self).__init__(name=name, start=start, end=end)
        if not fake:
            with tf.variable_scope(self._name):
                self.output = tf.nn.relu(inputs)
            self._tf_name = self.output.name.split("/")[0] + "/" + self.output.name.split("/")[1]
        else:
            assert isinstance(inputs, Fake)
            self.output = Fake(inputs.shape)
        self.description.append("ReLU")
        self.description.append(self.get_memory_footprint())

    def get_params(self):
        """Activation layers do not have parameters to return."""
        return []

    def get_memory_footprint(self):
        return 0
