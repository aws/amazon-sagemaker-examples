import logging

import numpy as np
import tensorflow as tf

from ..core import Fake, Layer
from .ops import get_param_from_name, load_pkl_obj


def _fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == "channels_first":
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


class Conv2DFixedPadding(Layer):
    """Definition for a convolution layer.
    Strided 2-D convolution with explicit padding.

    Args:
    Args:
        name: name of the module.
        inputs: the input symbol.
        start: layer number for begining.
        end: layer number for ending.
        weights: If need to initialize with some parameters, supply the numpy pickle path.
                 We will read the exact same layer name.
        data_format: typically 'channel_first'
        kernel_size: Size of each filter. Typically 3.
        filterS: Number of output channels required.
        strides: Stride of convolution.
    """

    def __init__(
        self,
        name,
        inputs,
        filters,
        kernel_size,
        strides,
        data_format,
        start=None,
        end=None,
        weights=None,
        weight_scope=None,
        fake=False,
    ):
        super(Conv2DFixedPadding, self).__init__(name=name, start=start, end=end)
        self.fake = fake
        if not self.fake:
            if weights is not None:
                params_name = weight_scope + "/" + str(name) + "/conv2d/"
                np_dict = load_pkl_obj(weights)
                kernel_np = np_dict[params_name + "kernel:0"]
                in_shp = inputs.shape.as_list()[1]
                if not kernel_np.shape[2] == in_shp:
                    kernel_np = np.resize(kernel_np, (kernel_size, kernel_size, in_shp, filters))
                kernel_initializer = tf.constant_initializer(kernel_np)
            else:
                kernel_initializer = tf.variance_scaling_initializer()
            with tf.variable_scope(self._name):
                if strides > 1:
                    inputs = _fixed_padding(inputs, kernel_size, data_format)

                self.output = tf.layers.conv2d(
                    inputs=inputs,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=("SAME" if strides == 1 else "VALID"),
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    data_format=data_format,
                )
            self._tf_name = self.output.name.split("/")[0] + "/" + self.output.name.split("/")[1]
        else:
            assert isinstance(inputs, Fake)
            in_shp = inputs.shape
            param_shp = (kernel_size, kernel_size, in_shp[1], filters)
            self.param = Fake(param_shp)
            if kernel_size > 1:
                out_height = int(np.floor(((in_shp[2] - kernel_size + 2) / strides)) + 1)
                out_width = int(np.floor(((in_shp[3] - kernel_size + 2) / strides)) + 1)
            else:
                out_height = in_shp[2] / strides
                out_width = in_shp[3] / strides
            out_shp = (None, filters, out_height, out_width)
            self.output = Fake(out_shp)
            self._tf_name = "fake"
        self.description.append("Conv")
        self.description.append(filters)
        self.description.append(kernel_size)
        self.description.append(strides)
        self.description.append(("SAME" if strides == 1 else "VALID"))
        self.description.append(self.get_memory_footprint())

    def _get_params_real(self):
        """Returns the kernel node"""
        return {self.get_name(): get_param_from_name(self._tf_name + "/conv2d/kernel:0")}

    def _get_memory_footprint_real(self):
        """Number of parameters in the layer"""
        params = self.get_params()[self.get_name()]
        return int(np.prod(params.shape))

    def _get_params_fake(self):
        return self.param

    def _get_memory_footprint_fake(self):
        return int(np.prod(self.get_params().shape))

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
