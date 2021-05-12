import tensorflow as tf

from ..core import Fake, Layer


class Pool(Layer):
    """Definition for a max pooling  layer.
    Strided 2-D pooling.

    Args:
    Args:
        name: name of the module.
        inputs: the input symbol.
        start: layer number for begining.
        end: layer number for ending.
        pool_size: Size of the pooling strides.
        strides: Strides of the pooling window.
        data_format: typically 'channel_first'
    """

    def __init__(
        self,
        name,
        inputs,
        pool_size,
        strides,
        padding,
        data_format,
        start=None,
        end=None,
        fake=False,
    ):
        super(Pool, self).__init__(name=name, start=start, end=end)
        if not fake:
            with tf.variable_scope(self._name):
                self.output = tf.layers.max_pooling2d(
                    inputs=inputs,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                )
            self._tf_name = self.output.name.split("/")[0] + "/" + self.output.name.split("/")[1]
        else:
            assert isinstance(inputs, Fake)
            self.output = Fake(inputs.shape)
        self.description.append("Pool")
        self.description.append(pool_size)
        self.description.append(strides)
        self.description.append(padding)
        self.description.append(self.get_memory_footprint())

    def get_params(self):
        """Pooling layers do not have parameters to return."""
        return []

    def get_memory_footprint(self):
        """Pooling layer has no memory"""
        return 0
