# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn.modules.pooling import MaxPool2d
from apex import amp

from maskrcnn_benchmark import NHWC

class UpSampleNearest2d_NHWC_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_size):
        ctx.output_size = output_size
        ctx.input_size = x.shape
        y = NHWC.upsample_nearest2d_cuda(x, output_size)
        # Need to save y as well
        return y 

    @staticmethod
    def backward(ctx, y_grad):
        input_size = ctx.input_size
        output_size = ctx.output_size

        return NHWC.upsample_nearest2d_backward_cuda(
                                   y_grad,
                                   output_size,
                                   input_size), None

class UpSampleNearest2d_NHWC(torch.nn.Module):
    def __init__(self, output_size):
        super(UpSampleNearest2d_NHWC, self).__init__()
        self.output_size = output_size
    def forward(self, x):
        return UpSampleNearest2d_NHWC_Impl.apply(x,
                                        self.output_size)

def upsample_nearest2d(x, output_size):
    op = UpSampleNearest2d_NHWC(output_size)
    return op(x)

