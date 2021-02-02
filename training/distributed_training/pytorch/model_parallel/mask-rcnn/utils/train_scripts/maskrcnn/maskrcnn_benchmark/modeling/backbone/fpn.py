# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers.nhwc import Conv2d_NHWC, nhwc_to_nchw_transform, nchw_to_nhwc_transform, interpolate_nhwc
from maskrcnn_benchmark.layers.nhwc import MaxPool2d_NHWC 
from maskrcnn_benchmark.layers.nhwc import init
from maskrcnn_benchmark import _C

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None, nhwc=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.nhwc = nhwc
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1, nhwc=nhwc)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1, nhwc=nhwc)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            interpolate_func = F.interpolate if not self.nhwc else interpolate_nhwc
            inner_top_down = interpolate_func(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1], self.nhwc)
            results.extend(last_results)
        if self.nhwc:
            results_nchw = []
            for i, f in enumerate(results):
                results_nchw.append(nhwc_to_nchw_transform(f))
            return [tuple(results), tuple(results_nchw)]
        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super(LastLevelMaxPool, self).__init__()
        self.max_pool_nhwc = MaxPool2d_NHWC(1,2,0)
        self.max_pool = nn.MaxPool2d(1,2,0)

    def forward(self, x, nhwc):
        op = self.max_pool_nhwc if nhwc else self.max_pool 
        return [op(x)] 


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels, nhwc):
        super(LastLevelP6P7, self).__init__()
        conv = conv2d_NHWC if nhwc else nn.Conv2d
        self.p6 = conv(in_channels, out_channels, 3, 2, 1)
        self.p7 = conv(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            init.kaiming_uniform_(module.weight, a=1, nhwc=nhwc)
            init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
