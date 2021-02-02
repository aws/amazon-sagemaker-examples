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

#include <ATen/cudnn/cudnn-wrapper.h>
#include <string>
#include <stdexcept>
#include <sstream>

struct THCState;

namespace at { namespace native { namespace nhwc {

class cudnn_exception : public std::runtime_error {
public:
  cudnnStatus_t status;
  cudnn_exception(cudnnStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  cudnn_exception(cudnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

inline void CUDNN_CHECK(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS) {
    if (status == CUDNN_STATUS_NOT_SUPPORTED) {
        throw cudnn_exception(status, std::string(cudnnGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    throw cudnn_exception(status, cudnnGetErrorString(status));
  }
}

inline void CUDA_CHECK(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::string msg("CUDA error: ");
    msg += cudaGetErrorString(error);
    throw std::runtime_error(msg);
  }
}

}}}  // namespace at::cudnn

namespace at { namespace native { namespace nchw {

class cudnn_exception : public std::runtime_error {
public:
  cudnnStatus_t status;
  cudnn_exception(cudnnStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  cudnn_exception(cudnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

inline void CUDNN_CHECK(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS) {
    if (status == CUDNN_STATUS_NOT_SUPPORTED) {
        throw cudnn_exception(status, std::string(cudnnGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    throw cudnn_exception(status, cudnnGetErrorString(status));
  }
}

inline void CUDA_CHECK(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::string msg("CUDA error: ");
    msg += cudaGetErrorString(error);
    throw std::runtime_error(msg);
  }
}

}}}  // namespace at::cudnn



