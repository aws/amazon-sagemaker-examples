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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <bitset>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>

int const threadsPerBlock = sizeof(unsigned long long) * 8;
int const block_size = sizeof(unsigned long long) * 8;
int const max_shmem_size = 49152; 

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

// initial_pos_mask is the initial positive masks for boxes, 1 if it is kept and 0 otherwise

__global__ void nms_reduce_batched(const int *n_boxes_arr, unsigned long long *dev_mask_cat, unsigned char *initial_pos_mask, 
                                   unsigned char *res_mask_byte_arr, int mask_dev_stride, int mask_res_stride) {
  int fmap_id = blockIdx.x;
  int tid = threadIdx.x;
  const int n_boxes = n_boxes_arr[fmap_id];
  int offset = 0;
  for (int i = 0; i < fmap_id; i++) offset += n_boxes_arr[i];
  const unsigned long long *dev_mask = dev_mask_cat + mask_dev_stride * fmap_id;
  initial_pos_mask += offset;
  unsigned char *res_mask_byte = res_mask_byte_arr + offset;
 
  const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
  //compute largest block we can fit in shared memory
  const unsigned int block_rows_max = max_shmem_size / sizeof(unsigned long long) / col_blocks - 1;
  //use intrinsics functions to compute largest block that is power of 2
  //power of 2 helps the main loop to be more efficient
  const unsigned int block_rows = 1 << (8 * sizeof(unsigned int) - __ffs(__brev(block_rows_max)));
  extern __shared__ unsigned long long mask_buf_sh[];
  unsigned long long *res_mask_sh = mask_buf_sh;
  unsigned long long *mask_block = mask_buf_sh + col_blocks; 
  for (int i = tid; i < col_blocks; i += blockDim.x) {
	  res_mask_sh[i] = 0;
          for (int j=0;j<8*sizeof(unsigned long long);j++){
	      if ( (i*64 +j) < n_boxes &&  (initial_pos_mask[i*64+j] == 0)) res_mask_sh[i] |= 1ULL<<j; 
	  }
  }

  __syncthreads();
  unsigned int *mask_block32 = (unsigned int*) mask_block ;
  unsigned int *res_mask_sh32 = (unsigned int*) res_mask_sh ;
  for (int i = 0; i < n_boxes; i += block_rows){
    int num_rows = min(n_boxes-i, block_rows);
    int block_max_elements = num_rows * col_blocks;
    for (int j = tid; j < block_max_elements; j += block_size) mask_block[j] = dev_mask[i * col_blocks + j];
    __syncthreads();
    int nblock = i / block_size;
    int num_rows_inner_loop;
    for (int k_start = 0; k_start < block_rows; k_start += block_size) {
      num_rows_inner_loop = min(num_rows, k_start + block_size) - k_start;
      for (int k = 0; k < num_rows_inner_loop; k++){
        if (!(res_mask_sh[nblock] & 1ULL << k)){
          for (int t = tid; t < col_blocks; t += block_size) 
            res_mask_sh[t] |= mask_block[(k + k_start) * col_blocks + t];
        }
        __syncthreads();
      }
      nblock++;
    }
  }
  for (int i = tid; i < n_boxes; i += block_size){
    int nblock = i / block_size;
    int in_block = i % block_size;
    res_mask_byte[i] = 1 - (unsigned char)((res_mask_sh[nblock] & 1ULL << in_block) >> in_block);
  }
}

__global__ void nms_kernel_batched(const int *n_boxes_arr, const float nms_overlap_thresh, const float *dev_boxes_cat, 
                                   unsigned long long *dev_mask_cat, int mask_stride) {
  const int fmap_id = blockIdx.z; 
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int n_boxes = n_boxes_arr[fmap_id];
  int offset = 0;
  for (int i = 0; i < fmap_id; i++) offset += n_boxes_arr[i];
  const float *dev_boxes = dev_boxes_cat + offset * 4;
  unsigned long long *dev_mask = dev_mask_cat + mask_stride * fmap_id;
  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
  if (row_size < 0 || col_size < 0) return;
  __shared__ float block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// initial_pos_mask is the initial positive masks for boxes, 1 if it is kept and 0 otherwise
at::Tensor nms_batched_cuda (const at::Tensor boxes_cat, const std::vector<int> n_boxes_vec, const at::Tensor n_boxes, const at::Tensor initial_pos_mask, float nms_overlap_thresh  ){
  //this function assumes input boxes are sorted
  //TODO: add an option for non sorted input boxes
  using scalar_t = float;
  AT_ASSERTM(boxes_cat.is_cuda(), "boxes_cat must be a CUDA tensor");
  AT_ASSERTM(n_boxes.is_cuda(), "n_boxes must be a CUDA tensor");
  int total_boxes=boxes_cat.size(0);
  int n_fmaps = n_boxes_vec.size();
  int n_boxes_max = *std::max_element(n_boxes_vec.begin(), n_boxes_vec.end());
  const int col_blocks_max = THCCeilDiv(n_boxes_max, threadsPerBlock);
  at::Tensor mask_dev_tensor = at::zeros({n_fmaps * n_boxes_max * col_blocks_max}, boxes_cat.options().dtype(at::kLong));
  unsigned long long *mask_dev = (unsigned long long*) mask_dev_tensor.data_ptr<int64_t>();
  at::Tensor keep = at::empty({total_boxes}, boxes_cat.options().dtype(at::kByte));
  dim3 blocks(THCCeilDiv(n_boxes_max, threadsPerBlock),
              THCCeilDiv(n_boxes_max, threadsPerBlock),
              n_fmaps);
  dim3 threads(threadsPerBlock);
  auto stream = at::cuda::getCurrentCUDAStream();
  nms_kernel_batched<<<blocks, threads, 0, stream.stream()>>>(n_boxes.data_ptr<int>(),
                                          nms_overlap_thresh,
                                          boxes_cat.data_ptr<scalar_t>(),
                                          mask_dev,
                                          n_boxes_max * col_blocks_max);   
                                          
  //last argument is the f_map stride in mask_dev array                                                                               
  nms_reduce_batched<<<n_fmaps, threadsPerBlock, max_shmem_size, stream.stream()>>>(n_boxes.data_ptr<int>(), 
                                                                                    mask_dev, 
										    initial_pos_mask.data_ptr<unsigned char>(),
                                                                                    keep.data_ptr<unsigned char>(), 
                                                                                    n_boxes_max * col_blocks_max, 
                                                                                    col_blocks_max); 
                                                                  
  return keep;
}

