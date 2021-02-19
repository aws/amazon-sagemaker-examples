/**
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
 */

#pragma once
#include <torch/extension.h>
#ifndef _nhwc_h_
#define _nhwc_h_ 

namespace at { namespace native { namespace nhwc {
// NHWC conv
// fprop (X, W) -> Y

at::Tensor cudnnNhwcToNchw(const at::Tensor& input);
at::Tensor cudnnNchwToNhwc(const at::Tensor& input);
at::Tensor cudnn_convolution_nhwc(
    const at::Tensor& input_t, const at::Tensor& weight_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic);
// fprop (X, W, b) -> Y
at::Tensor cudnn_convolution_with_bias_nhwc(
    const at::Tensor& input_t, const at::Tensor& weight_t, const at::Tensor& bias_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic);
// bprop (X, dY, W) -> (dX, dW)
std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask);
// bprop (X, dY, W) -> (dX, dW, db)
std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_convolution_backward_with_bias_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask);
std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_convolution_transpose_backward_with_bias_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask);
std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask);
at::Tensor cudnn_convolution_transpose_nhwc(
    const at::Tensor& input_t, const at::Tensor& weight_t,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic);
at::Tensor cudnn_convolution_transpose_with_bias_nhwc(
    const at::Tensor& input_t, const at::Tensor& weight_t, const at::Tensor& bias_t,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic);

at::Tensor upsample_nearest2d_cuda(const at::Tensor& input, std::vector<long> output_size); 
at::Tensor upsample_nearest2d_backward_cuda(const at::Tensor& grad_output, std::vector<long> output_size, std::vector<long> input_size);

}}}

// NHWC MaxPool
at::Tensor max_pool_nhwc_fwd(
                       const at::Tensor& x,
                       const int kernel,
                       const int stride,
                       const int padding,
                       const int dilation);

at::Tensor max_pool_nhwc_bwd(const at::Tensor& x,
                             const at::Tensor& y,
                             const at::Tensor& grad_y,
                             const int kernel,
                             const int stride,
                             const int padding,
                             const int dilation);

#endif
