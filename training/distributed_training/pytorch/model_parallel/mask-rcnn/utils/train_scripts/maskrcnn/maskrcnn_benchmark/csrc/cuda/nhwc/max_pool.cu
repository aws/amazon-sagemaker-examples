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
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Handle.h>
//#include <THC/THCNumerics.cuh>

#include "Descriptors.h"

#include <cuda.h>
#include <cudnn.h>

at::Tensor max_pool_nhwc_fwd(
                       const at::Tensor& x_t,
                       const int kernel,
                       const int stride,
                       const int padding,
                       const int dilation) {
  // we assume later x is contiguous
  at::Tensor x = x_t.contiguous();

  // dimensions
  const int N = x.size(0);
  const int C = x.size(3);
  const int H = x.size(1);
  const int W = x.size(2);

  // Create the cudnn pooling desc
  cudnnPoolingDescriptor_t pool_desc;
  cudnnCreatePoolingDescriptor(&pool_desc);

  int kernelA[] = {kernel, kernel};
  int padA[] = {padding, padding};
  int strideA[] = {stride, stride};
  cudnnSetPoolingNdDescriptor(pool_desc, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_PROPAGATE_NAN,
                            2, kernelA, padA, strideA);

  at::native::nhwc::TensorDescriptor x_desc, y_desc;
  x_desc.set(x);

  // Get the output dimensions
  int outputDimA[4];
  cudnnGetPoolingNdForwardOutputDim(pool_desc, x_desc.desc(), 4, outputDimA);

  // at::Tensor y = x.type().tensor({outputDimA[0], outputDimA[2], outputDimA[3], outputDimA[1]});
  at::Tensor y = at::empty({outputDimA[0], outputDimA[2], outputDimA[3], outputDimA[1]}, x.options());
  y_desc.set(y);

  float alpha = 1., beta = 0.;
  auto handle = at::native::getCudnnHandle();
  cudnnPoolingForward(handle,
                      pool_desc,
                      &alpha,
                      x_desc.desc(),
                      x.data<at::Half>(),
                      &beta,
                      y_desc.desc(),
                      y.data<at::Half>());

  cudnnDestroyPoolingDescriptor(pool_desc);
  return y;
}

at::Tensor max_pool_nhwc_bwd(const at::Tensor& x_t,
                             const at::Tensor& y_t,
                             const at::Tensor& grad_y_t,
                             const int kernel,
                             const int stride,
                             const int padding,
                             const int dilation) {
  // we assume later x, y and grad_y are contiguous
  at::Tensor x = x_t.contiguous();
  at::Tensor y = y_t.contiguous();
  at::Tensor grad_y = grad_y_t.contiguous();

  // dimensions
  const int N = x.size(0);
  const int C = x.size(3);
  const int H = x.size(1);
  const int W = x.size(2);

  // Create the cudnn pooling desc
  cudnnPoolingDescriptor_t pool_desc;
  cudnnCreatePoolingDescriptor(&pool_desc);

  int kernelA[] = {kernel, kernel};
  int padA[] = {padding, padding};
  int strideA[] = {stride, stride};
  cudnnSetPoolingNdDescriptor(pool_desc, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_PROPAGATE_NAN,
                            2, kernelA, padA, strideA);

  at::native::nhwc::TensorDescriptor x_desc, y_desc, dx_desc, dy_desc;
  x_desc.set(x);
  y_desc.set(y);
  dy_desc.set(grad_y);

  at::Tensor dx = at::empty_like(x);
  dx_desc.set(dx);

  float alpha = 1., beta = 0.;
  auto handle = at::native::getCudnnHandle();

  cudnnPoolingBackward(handle,
                       pool_desc,
                       &alpha,
                       y_desc.desc(),
                       y.data<at::Half>(),
                       dy_desc.desc(),
                       grad_y.data<at::Half>(),
                       x_desc.desc(),
                       x.data<at::Half>(),
                       &beta,
                       dx_desc.desc(),
                       dx.data<at::Half>());

  return dx;
}

