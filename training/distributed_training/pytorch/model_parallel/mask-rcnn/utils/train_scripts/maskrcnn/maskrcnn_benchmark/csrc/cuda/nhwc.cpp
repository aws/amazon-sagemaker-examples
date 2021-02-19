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

#include "nhwc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cudnn_convolution_nhwc", &at::native::nhwc::cudnn_convolution_nhwc, "cudnn_convolution_nhwc");
  m.def("cudnn_convolution_transpose_nhwc", &at::native::nhwc::cudnn_convolution_transpose_nhwc, "cudnn_convolution_transpose_nhwc");
  m.def("cudnn_convolution_transpose_backward_nhwc", &at::native::nhwc::cudnn_convolution_transpose_backward_nhwc, "cudnn_convolution_transpose_backward_nhwc");
  m.def("cudnn_convolution_transpose_with_bias_nhwc", &at::native::nhwc::cudnn_convolution_transpose_with_bias_nhwc, "cudnn_convolution_transpose_nhwc");
  m.def("cudnn_convolution_transpose_backward_with_bias_nhwc", &at::native::nhwc::cudnn_convolution_transpose_backward_with_bias_nhwc, "cudnn_convolution_transpose_backward_nhwc");
  m.def("cudnn_convolution_with_bias_nhwc", &at::native::nhwc::cudnn_convolution_with_bias_nhwc, "cudnn_convolution_with_bias_nhwc");
  m.def("cudnn_convolution_backward_nhwc", &at::native::nhwc::cudnn_convolution_backward_nhwc, "cudnn_convolution_backward_nhwc");
  m.def("cudnn_convolution_backward_with_bias_nhwc", &at::native::nhwc::cudnn_convolution_backward_with_bias_nhwc, "cudnn_convolution_backward_with_bias_nhwc");
  m.def("cudnnNhwcToNchw", &at::native::nhwc::cudnnNhwcToNchw, "cudnnNhwcToNchw");
  m.def("cudnnNchwToNhwc", &at::native::nhwc::cudnnNchwToNhwc, "cudnnNhcwToNhwc");
  m.def("upsample_nearest2d_cuda", &at::native::nhwc::upsample_nearest2d_cuda, "upsample_nearest2d_cuda");
  m.def("upsample_nearest2d_backward_cuda", &at::native::nhwc::upsample_nearest2d_backward_cuda, "upsample_nearest2d_backward_cuda");
  // MaxPool
  m.def("max_pool_fwd_nhwc", &max_pool_nhwc_fwd, "max_pool_fwd_nhwc");
  m.def("max_pool_bwd_nhwc", &max_pool_nhwc_bwd, "max_pool_bwd_nhwc");
}
