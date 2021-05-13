import numpy as np
import tensorflow as tf

from ..core import Fake, Layer
from .ops import get_param_from_name, load_pkl_obj


class Dense(Layer):
    """Definition for a dense layer.

    Args:
    Args:
        name: name of the module.
        inputs: the input symbol.
        start: layer number for begining.
        end: layer number for ending.
        weights: If need to initialize with some parameters, supply the numpy pickle path.
                 We will read the exact same layer name.
        units: Number of neurons in the layer.
    """

    def __init__(
        self, name, inputs, units, start=None, end=None, weights=None, weight_scope=None, fake=False
    ):
        super(Dense, self).__init__(name=name, start=start, end=end)
        self.fake = fake
        if not self.fake:
            if weights is not None:
                params_name = weight_scope + "/" + str(name) + "/dense/"
                np_dict = load_pkl_obj(weights)
                kernel_np = np_dict[params_name + "kernel:0"]
                bias_np = np_dict[params_name + "bias:0"]
                in_shp = inputs.shape.as_list()[1]
                if not kernel_np.shape[1] == in_shp:
                    kernel_np = np.resize(kernel_np, (in_shp, units))
                kernel_initializer = tf.constant_initializer(kernel_np)
                bias_initializer = tf.constant_initializer(bias_np)
            else:
                kernel_initializer = None
                bias_initializer = tf.zeros_initializer()
            with tf.variable_scope(self._name):
                self.output = tf.layers.dense(
                    inputs=inputs,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    units=units,
                )
            self._tf_name = self.output.name.split("/")[0] + "/" + self.output.name.split("/")[1]
        else:
            assert isinstance(inputs, Fake)
            in_shp = inputs.shape[1]
            self.param = Fake((in_shp, units))
            self.output = Fake((None, units))
        self.description.append("Dense")
        self.description.append(units)
        self.description.append(self.get_memory_footprint())

    def _get_params_real(self):
        """Returns the kernel node"""
        kernel = get_param_from_name(self._tf_name + "/dense/kernel:0")
        bias = get_param_from_name(self._tf_name + "/dense/bias:0")
        return {self.get_name(): [kernel, bias]}

    def _get_memory_footprint_real(self):
        """Returns the number of paramters"""
        params = self.get_params()[self.get_name()]
        return int(np.prod(params[0].shape)) + int(np.prod(params[1].shape))

    def _get_memory_footprint_fake(self):
        return int(np.prod(self.get_params().shape))

    def _get_params_fake(self):
        return self.param

    def get_memory_footprint(self):
        if self.fake:
            return self._get_memory_footprint_fake()
        else:
            return self._get_memory_footprint_real()

    def get_params(self):
        if self.fake:
            return self._get_params_fake()
        else:
            return self._get_params_real()
