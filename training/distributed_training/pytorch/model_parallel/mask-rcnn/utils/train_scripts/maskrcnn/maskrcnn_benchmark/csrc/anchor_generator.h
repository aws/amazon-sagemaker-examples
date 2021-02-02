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

#pragma once
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <THC/THC.h>
#include <vector>

std::vector<at::Tensor> anchor_generator(std::vector<int64_t> image_shape,       // (height, width)
                                        std::vector<int64_t> feature_map_size,   // (height, width)
                                        at::Tensor& cell_anchors,                // shape: [1, 3, 4]
                                        const int stride,
                                        const float straddle_thresh);

