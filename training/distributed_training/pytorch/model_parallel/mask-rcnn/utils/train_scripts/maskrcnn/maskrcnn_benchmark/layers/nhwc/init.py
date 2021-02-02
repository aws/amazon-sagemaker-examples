# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from torch import nn
from torch.nn import functional as F

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', nhwc=False):
    if nhwc:
        weight_tensor_nchw = tensor.data.permute(0,3,1,2).contiguous()
    else:
        weight_tensor_nchw = tensor
    nn.init.kaiming_uniform_(weight_tensor_nchw, a=a, mode=mode, nonlinearity=nonlinearity)
    if nhwc:
        tensor.data.copy_(weight_tensor_nchw.permute(0,2,3,1).contiguous())

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu', nhwc=False):
    if nhwc:
        weight_tensor_nchw = tensor.data.permute(0,3,1,2).contiguous()
    else:
        weight_tensor_nchw = tensor
    nn.init.kaiming_normal_(weight_tensor_nchw, a=a, mode=mode, nonlinearity=nonlinearity)
    if nhwc:
        tensor.data.copy_(weight_tensor_nchw.permute(0,2,3,1).contiguous())

def normal_(tensor, mean=0.0, std=1.0, nhwc=False):
    if nhwc:
        weight_tensor_nchw = tensor.data.permute(0,3,1,2).contiguous()
    else:
        weight_tensor_nchw = tensor
    nn.init.normal_(weight_tensor_nchw, mean=mean, std=std)
    if nhwc:
        tensor.data.copy_(weight_tensor_nchw.permute(0,2,3,1).contiguous())

def constant_(tensor, val, nhwc=False):
    if nhwc and len(tensor.shape) == 4:
        weight_tensor_nchw = tensor.data.permute(0,3,1,2).contiguous()
    else:
        weight_tensor_nchw = tensor
    nn.init.constant_(weight_tensor_nchw, val=val)
    if nhwc and len(tensor.shape) == 4:
        tensor.data.copy_(weight_tensor_nchw.permute(0,2,3,1).contiguous())


