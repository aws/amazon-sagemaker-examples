/******************************************************************************
*
* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*

 ******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/Exceptions.h>
#include <cudnn.h>
#include "THC/THC.h"
#include "torch/torch.h"
#include <ATen/cudnn/cudnn-wrapper.h>
#include "Descriptors.h"
//#include <ATen/cudnn/Types.h>
#include "Types.h"
#include <ATen/cudnn/Utils.h>
#include "ParamsHash.h"

#include <ATen/TensorUtils.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
namespace at { namespace native { namespace nhwc {

at::Tensor cudnnNhwcToNchw(const at::Tensor& input) {

  int N = input.size(0);
  int C = input.size(3);
  int H = input.size(1);
  int W = input.size(2);
  at::Tensor output = at::empty({N,C,H,W}, input.options());
  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(input);
  at::native::nhwc::TensorDescriptor in_desc;
  at::native::nchw::TensorDescriptor out_desc;
  in_desc.set(input);
  out_desc.set(output);
  float alpha=1.0f;
  float beta=0.0f;
  cudnnTransformTensor(handle, &alpha, in_desc.desc(), input.data_ptr(), &beta, out_desc.desc(), output.data_ptr());
  return output;
}

at::Tensor cudnnNchwToNhwc(const at::Tensor& input) {

  int N = input.size(0);
  int C = input.size(1);
  int H = input.size(2);
  int W = input.size(3);
  at::Tensor output = at::empty({N,H,W,C}, input.options());
  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(input);
  at::native::nchw::TensorDescriptor in_desc;
  at::native::nhwc::TensorDescriptor out_desc;
  in_desc.set(input);
  out_desc.set(output);
  float alpha=1.0f;
  float beta=0.0f;
  cudnnTransformTensor(handle, &alpha, in_desc.desc(), input.data_ptr(), &beta, out_desc.desc(), output.data_ptr());
  return output;
}
}}}
