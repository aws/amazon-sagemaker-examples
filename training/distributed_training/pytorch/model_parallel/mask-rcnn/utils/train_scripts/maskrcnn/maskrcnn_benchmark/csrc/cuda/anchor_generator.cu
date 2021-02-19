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

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCNumerics.cuh>
#include <THC/THC.h>
#include <cuda.h>
#include <vector>

__device__
float4 add_boxes(const float4& a, const float4& b) {
  return float4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

/**
 * Essentially bolis down to a grid-stride loop
 * - get the index of the output and backtrack what its values should
 *   be based on the arange that would have been created.
 * - Easy parallelism - use (BSZ_X, A) as block dimensions to parallelize over
 *   the A anchors used.
 * - Accesses to global memory are all done via. float4
 */
__global__
void generate_anchors_single(const int image_height,
                             const int image_width,
                             const int feature_height,
                             const int feature_width,
                             const float4* anchor_data, // [1, 3, 4]
                             const int stride,
                             const int A,
                             float4 *anchors,
                             const float straddle_thresh,
                             uint8_t* inds_inside) {

  // size of arange is floor(start - end / step)
  // in this case, floor((feature{height,width} * stride - 0) / stride)
  const int len_x = (int)floorf(feature_width);
  const int len_y = (int)floorf(feature_height);

#if 0
  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    printf("len_x: %d, len_y: %d\n", len_x, len_y);
  }
#endif
  // Standard grid-stride loop over output size
  for (int output_idx = threadIdx.x + blockIdx.x * blockDim.x;
           output_idx < len_x * len_y;
           output_idx += gridDim.x * blockDim.x) {
    // local box is (xp, yp, xp, yp)
    // where xp = x[output_idx % len(x)]
    //       yp = y[output_idx / len(y)]
    // and x = (output_idx % len(x)) * step
    //     y = (output_idx / len(y)) * step
    const float x = (output_idx % len_x) * stride;
    const float y = (output_idx / len_x) * stride;

    // This is the basic box
    float4 box{x, y, x, y};

    // parallelize over anchors
    const int i = threadIdx.y;
    // for (int i = 0; i < A; ++i) {
    const float4 a = anchor_data[i];

    float4 tmp = add_boxes(box, a);

    anchors[output_idx * A + i] = tmp;

    // for each anchor, now check
    if (straddle_thresh >= 0.f) {
      inds_inside[output_idx * A + i] = (tmp.x >= -straddle_thresh)
                                      & (tmp.y >= -straddle_thresh)
                                      & (tmp.z < image_width + straddle_thresh)
                                      & (tmp.w < image_height + straddle_thresh);
    } else {
      inds_inside[output_idx * A + i] = 1;
    }
  }
}


std::vector<at::Tensor> anchor_generator(
    std::vector<int64_t> image_shape,       // (height, width)
    std::vector<int64_t> feature_map_size,  // (height, width)
    at::Tensor& cell_anchors,               // shape: [1, 3, 4]
    const int stride,
    const float straddle_thresh) {

  // Need to work out some sizes for the kernel
  const float h_start = 0.;
  const float h_end = feature_map_size[0] * stride;
  const int h_elems = (int)std::floor( (h_end - h_start) / stride );

  const float w_start = 0., w_end = feature_map_size[1] * stride;
  const int w_elems = (int)std::floor( (w_end - w_start) / stride );

  // If cell anchors are [A, 4]
  const int A = cell_anchors.size(0);
  // output anchors are h_elems * w_elems * A * 4 values, so allocate that now.
  at::Tensor anchors = torch::zeros({h_elems * w_elems * A, 4}, torch::CUDA(at::kFloat));
  // also output a bool map of anchors being inside the image
  at::Tensor inds_inside = torch::zeros({h_elems * w_elems * A}, torch::CUDA(at::kByte));

  // CUDA grid is going to be (32, A) * (h_elems * w_elems / 32)
  const int blockx = 64;
  dim3 block(blockx, A);
  dim3 grid((h_elems * w_elems + (blockx - 1)) / blockx);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  generate_anchors_single<<<grid, block, 0, stream>>>(
                             image_shape[0],
                             image_shape[1],
                             feature_map_size[0],
                             feature_map_size[1],
                             reinterpret_cast<float4*>(cell_anchors.data_ptr<float>()),
                             stride,
                             A,
                             reinterpret_cast<float4*>(anchors.data_ptr<float>()),
                             straddle_thresh,
                             inds_inside.data_ptr<uint8_t>());
  THCudaCheck(cudaGetLastError());

  return {anchors, inds_inside};
}

