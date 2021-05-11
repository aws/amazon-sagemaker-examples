import logging

import numpy as np

from ..core import Module
from . import BatchNorm, Conv, ReLU


class BuildingBlock(Module):
    """A single block for ResNet v2, without a bottleneck.

    Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').
        layer_count: What is the number of layer at which we are in the building
            of the resnet.
        remove_layers: What layers (relative to the layer_count) do we want to
            remove? send a boolean array.

    Note:
        remove_layers is ignored if strides > 1
    """

    def __init__(
        self,
        name,
        inputs,
        filters,
        training,
        projection_shortcut,
        strides,
        data_format,
        layer_count=0,
        remove_layers=[False] * 152,
        weights=None,
        weight_scope=None,
        fake=False,
    ):
        super(BuildingBlock, self).__init__(name=name)
        begin_layer = layer_count
        self.description = ["BuildingBlock"]

        shortcut = inputs
        self.layer_count = layer_count

        if not remove_layers[self.layer_count]:
            l = BatchNorm(
                name=str(self.layer_count),
                inputs=inputs,
                training=training,
                weights=weights,
                weight_scope=weight_scope,
                data_format=data_format,
                start=self.layer_count,
                fake=fake,
                end=self.layer_count,
            )
            self.add_module(l, l.get_name())
            inputs = l.output
        self.layer_count += 1

        if not remove_layers[self.layer_count]:
            l = ReLU(
                name=str(self.layer_count),
                inputs=inputs,
                start=self.layer_count,
                fake=fake,
                end=self.layer_count,
            )
            inputs = l.output
            self.add_module(l, l.get_name())
        self.layer_count += 1

        if projection_shortcut is not None:
            shortcut_inputs = inputs

        if not remove_layers[self.layer_count]:
            l = Conv(
                name=str(self.layer_count),
                inputs=inputs,
                filters=filters,
                kernel_size=3,
                weights=weights,
                weight_scope=weight_scope,
                strides=strides,
                data_format=data_format,
                start=self.layer_count,
                fake=fake,
                end=self.layer_count,
            )
            inputs = l.output
            self.add_module(l, l.get_name())
        self.layer_count += 1

        if not remove_layers[self.layer_count]:
            l = BatchNorm(
                name=str(self.layer_count),
                inputs=inputs,
                training=training,
                weights=weights,
                weight_scope=weight_scope,
                data_format=data_format,
                fake=fake,
                start=self.layer_count,
                end=self.layer_count,
            )
            inputs = l.output
            self.add_module(l, l.get_name())
        self.layer_count += 1

        if not remove_layers[self.layer_count]:
            l = ReLU(
                name=str(self.layer_count),
                inputs=inputs,
                start=self.layer_count,
                fake=fake,
                end=self.layer_count,
            )
            inputs = l.output
            self.add_module(l, l.get_name())
        self.layer_count += 1

        if not remove_layers[self.layer_count]:
            l = Conv(
                name=str(self.layer_count),
                inputs=inputs,
                filters=filters,
                kernel_size=3,
                strides=1,
                weights=weights,
                weight_scope=weight_scope,
                data_format=data_format,
                start=self.layer_count,
                fake=fake,
                end=self.layer_count,
            )
            inputs = l.output
            self.add_module(l, l.get_name())
        self.layer_count += 1

        # If all conv layers are removed, remove the projection shortcut also.
        if projection_shortcut:
            if not remove_layers[begin_layer + 2]:
                strides_ps = strides
            else:
                strides_ps = 1

            if remove_layers[begin_layer + 2] and remove_layers[begin_layer + 5]:
                projection_shortcut = None

        if projection_shortcut is not None:
            shortcut_module, self.layer_count = projection_shortcut(
                inputs=shortcut_inputs,
                layer_count_ps=self.layer_count,
                strides_ps=strides_ps,
                weight_scope=weight_scope,
                fake=fake,
                weights=weights,
            )
            shortcut = shortcut_module.output
            self.add_module(shortcut_module, shortcut_module.get_name())
            if fake:
                assert inputs.shape == shortcut.shape
                self.output = shortcut
            else:
                self.output = inputs + shortcut
        else:
            self.layer_count += 1
            self.output = inputs

    def get_layer_count(self):
        return self.layer_count
