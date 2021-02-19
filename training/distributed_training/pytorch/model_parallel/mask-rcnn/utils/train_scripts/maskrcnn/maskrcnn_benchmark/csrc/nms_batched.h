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
#include "cuda/vision.h"
#ifndef _nms_batched_h_
#define _nms_batched_h_ 

// initial_pos_mask is the initial positive masks for boxes, 1 if it is kept and 0 otherwise
at::Tensor nms_batched(const at::Tensor boxes_cat,
                                const std::vector<int> n_boxes_vec, 
                                const at::Tensor n_boxes, const at::Tensor initial_pos_mask,
                                const float nms_overlap_thresh){
  at::Tensor result = nms_batched_cuda(boxes_cat, 
                                n_boxes_vec, 
                                n_boxes, initial_pos_mask,
                                nms_overlap_thresh);
  return result;

}                                 

#endif

