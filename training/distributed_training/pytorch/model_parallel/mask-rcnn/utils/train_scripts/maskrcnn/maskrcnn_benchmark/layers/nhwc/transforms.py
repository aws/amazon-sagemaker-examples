# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
from apex import amp
from maskrcnn_benchmark import NHWC
import smdistributed.modelparallel.torch as smp

class NHWCToNCHW_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        torch.cuda.set_device(smp.local_rank())
        y = NHWC.cudnnNhwcToNchw(x)
        return y
 
    @staticmethod
    def backward(ctx, y_grad):
        torch.cuda.set_device(smp.local_rank())
        x_grad = NHWC.cudnnNchwToNhwc(y_grad)
        return x_grad

class NCHWToNHWC_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        torch.cuda.set_device(smp.local_rank())
        y = NHWC.cudnnNchwToNhwc(x)
        return y
 
    @staticmethod
    def backward(ctx, y_grad):
        torch.cuda.set_device(smp.local_rank())
        x_grad = NHWC.cudnnNhwcToNchw(y_grad)
        return x_grad

class NHWCToNCHW(torch.nn.Module):
    def __init__(self):
        super(NHWCToNCHW, self).__init__()
    def forward(self, x):
        return NHWCToNCHW_Impl.apply(x)

class NCHWToNHWC(torch.nn.Module):
    def __init__(self):
        super(NCHWToNHWC, self).__init__()
    def forward(self, x):
        return NCHWToNHWC_Impl.apply(x)

