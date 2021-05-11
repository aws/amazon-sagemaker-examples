# This file puts a wrapper on top of all tf batch norm layer.
import numpy as np
import tensorflow as tf

from ..core import Fake, Layer
from .ops import get_param_from_name, load_pkl_obj

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class BatchNorm(Layer):
    """Definition for a batch norm layer.

    Args:
        name: name of the module.
        inputs: the input symbol.
        start: layer number for begining.
        end: layer number for ending.
        weights: If need to initialize with some parameters, supply the numpy pickle path.
                 We will read the exact same layer name.
        data_format: typically 'channel_first'
        training: Boolean, False means, the BatchNorm will run as inference mode, True means it
                  will update the running mean and variance.
    """

    def __init__(
        self,
        name,
        inputs,
        training,
        data_format,
        start=None,
        end=None,
        weights=None,
        weight_scope=None,
        fake=False,
    ):
        super(BatchNorm, self).__init__(name=name, start=start, end=end)
        self.fake = fake
        if not self.fake:
            if weights is not None:
                params_name = weight_scope + "/" + str(name) + "/batch_normalization/"
                np_dict = load_pkl_obj(weights)
                beta_np = np_dict[params_name + "beta:0"]
                gamma_np = np_dict[params_name + "gamma:0"]
                moving_mean_np = np_dict[params_name + "moving_mean:0"]
                moving_variance_np = np_dict[params_name + "moving_variance:0"]
                in_shp = inputs.shape.as_list()[1]
                if not beta_np.shape[0] == in_shp:
                    beta_np = np.resize(beta_np, (in_shp,))
                    gamma_np = np.resize(gamma_np, (in_shp,))
                    moving_mean_np = np.resize(moving_mean_np, (in_shp))
                    moving_variance_np = np.resize(moving_variance_np, (in_shp))
                beta_initializer = tf.constant_initializer(beta_np)
                gamma_initializer = tf.constant_initializer(gamma_np)
                moving_mean_initializer = tf.constant_initializer(moving_mean_np)
                moving_variance_initializer = tf.constant_initializer(moving_variance_np)
            else:
                beta_initializer = tf.zeros_initializer()
                gamma_initializer = tf.ones_initializer()
                moving_mean_initializer = tf.zeros_initializer()
                moving_variance_initializer = tf.ones_initializer()
            with tf.variable_scope(self._name):
                self.output = tf.layers.batch_normalization(
                    inputs=inputs,
                    axis=1 if data_format == "channels_first" else 3,
                    momentum=_BATCH_NORM_DECAY,
                    epsilon=_BATCH_NORM_EPSILON,
                    center=True,
                    scale=True,
                    training=training,
                    beta_initializer=beta_initializer,
                    gamma_initializer=gamma_initializer,
                    moving_mean_initializer=moving_mean_initializer,
                    moving_variance_initializer=moving_variance_initializer,
                    fused=True,
                )
            self._tf_name = self.output.name.split("/")[0] + "/" + self.output.name.split("/")[1]
        else:
            assert isinstance(inputs, Fake)
            self.output = Fake(inputs.shape)
            self.param = Fake(inputs.shape[1] * 4)
        self.description.append("BatchNorm")
        self.description.append(self.get_memory_footprint())

    def _get_params_real(self):
        """Returns a list of [gamma, beta]"""
        gamma = get_param_from_name(self._tf_name + "/batch_normalization/gamma:0")
        beta = get_param_from_name(self._tf_name + "/batch_normalization/beta:0")
        moving_mean = get_param_from_name(self._tf_name + "/batch_normalization/moving_mean:0")
        moving_variance = get_param_from_name(
            self._tf_name + "/batch_normalization/moving_variance:0"
        )
        return {self.get_name(): [gamma, beta, moving_mean, moving_variance]}

    def _get_memory_footprint_real(self):
        """Returns the number of parameters in the layer"""
        params = self.get_params()[self.get_name()]
        gamma = params[0]
        beta = params[1]
        moving_mean = params[2]
        moving_variance = params[3]
        return (
            int(np.prod(gamma.shape))
            + int(np.prod(beta.shape))
            + int(np.prod(moving_mean.shape))
            + int(np.prod(moving_variance.shape))
        )

    def _get_memory_footprint_fake(self):
        return int(self.get_params().shape)

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
