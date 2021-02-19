# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from maskrcnn_benchmark import _C
from maskrcnn_benchmark import NHWC

from apex import amp

class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, is_nhwc):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.is_nhwc = is_nhwc
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, is_nhwc
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        if not ctx.is_nhwc:
            bs, ch, h, w = ctx.input_shape
        else:
            bs, h, w, ch = ctx.input_shape
        ## TODO: NHWC kernel + transposes is faster than NCHW backward kernel
        ## Might change to transposes + NHWC kernel if we want to speed up NCHW case
        ## Cast to fp32 for the kernel because FP16 atomics is slower than FP32 in Volta
        grad_input = _C.roi_align_backward(
            grad_output.float(),
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
            ctx.is_nhwc
        ).half()
        return grad_input, None, None, None, None, None


roi_align = _ROIAlign.apply

class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, is_nhwc):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.nhwc = is_nhwc

    def forward(self, input, rois):
        return roi_align(
            input, rois.float(), self.output_size, self.spatial_scale, self.sampling_ratio, self.nhwc
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
