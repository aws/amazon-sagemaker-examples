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

// TODO: Go through all the checking code again and make sure
// we haven't missed anything.

// ---------------------------------------------------------------------
//
// Math
//
// ---------------------------------------------------------------------

//cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
//  if (tensor.scalar_type() == at::kFloat) {
//    return CUDNN_DATA_FLOAT;
//  } else if (tensor.scalar_type() == at::kDouble) {
//    return CUDNN_DATA_DOUBLE;
//  } else if (tensor.scalar_type() == at::kHalf) {
//    return CUDNN_DATA_HALF;
//  }
//  std::string msg("getCudnnDataType() not supported for ");
//  msg += toString(tensor.scalar_type());
//  throw std::runtime_error(msg);
//}


constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 3;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 3;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 3;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

// NB: conv_output_size and conv_input_size are not bijections,
// as conv_output_size loses information; this is why conv_input_size
// takes an extra output_padding argument to resolve the ambiguity.

std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

// Handle [N, H, W, C] format -- adjust offsets into pad, stride, dilation, etc.
std::vector<int64_t> conv_output_size_nhwc(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[3] = weight_size[weight_output_channels_dim];
  for (size_t d = 1; d < dim-1; ++d) {
    auto kernel = dilation[d - 1] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 1])
                        - kernel) / stride[d - 1] + 1;
  }
  return output_size;
}

std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[3] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 1; d < dim-1; ++d) {
    int kernel = dilation[d - 1] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 1] - (2 * padding[d - 1]) +
                     kernel + output_padding[d - 1];
  }
  return input_size;
}

// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// Note [Legacy CuDNN grouped convolution support]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CuDNN earlier than CuDNN 7 does not directly support group
// convolution, so we provide support for it by sequentially
// running a convolution per group  with appropriately
// adjusted sizes.  https://blog.yani.io/filter-group-tutorial/
// has a fairly good diagram explaining how it works.

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  if (args.size() > expected_size){
    std::stringstream ss;
    ss << "Too many " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
  else if (args.size() < expected_size){
    std::stringstream ss;
    ss << "Not enough " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
}


// NOTE [ Convolution checks ]
//
// NB: For many call sites, it is not strictly necessary to check all of
// these relationships (for example, for forward convolution, we compute
// the size of output ourselves, so we don't actually need to check
// output.  However, writing a single function that does everything
// means we get to reuse it for both forwards and all backwards
// variants, even when the set of "real" inputs varies.  The magic of
// relational computing!
//
// (There is one downside, which is that it is slightly harder to write
// error messages which are able to distinguish between real inputs
// (which the user can change) and computed inputs (which the user can
// only indirectly affect).  It would be an interesting exercise to
// come up with a general framework to handle such situations.)
static void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

// Handle explicit [N, H, W, C] ordering
static void convolution_shape_check_nhwc(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  int input_stride[2 + max_dim];
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool deterministic) {

  cudnnDataType_t dataType = getCudnnDataType(input);
  memset(params, 0, sizeof(ConvolutionParams));
  params->dataType = dataType;
  // ASSERT(weight.dim() == input.dim())
  for (int i = 0; i != input.dim(); ++i) {
    params->input_size[i] = (int) input.size(i);
    params->input_stride[i] = (int) input.stride(i);
    params->weight_size[i] = (int) weight.size(i);
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
}

// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  cudnnHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  const Tensor& input, output, weight;
  ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  }
};

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// TODO: Use something less heavy duty than a big honking mutex
template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<ConvolutionParams, T, ParamsHash<ConvolutionParams>, ParamsEqual<ConvolutionParams>> map;

  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }
};

BenchmarkCache<cudnnConvolutionFwdAlgo_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgo_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgo_t> bwd_filter_algos;

// TODO: Stop manually allocating CUDA memory; allocate an ATen byte
// tensor instead.
struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    data = THCudaMalloc(globalContext().lazyInitCUDA(), size);
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(globalContext().lazyInitCUDA(), data);
    }
  }

  size_t size;
  void* data;
};

template<typename algo_t>
struct algorithm_search {
};

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionFwdAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        sz
    );
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        sz);
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        sz);
}

template<typename algo_t>
size_t getMaxWorkspaceSize(
    const ConvolutionArgs& args,
    const algo_t *algo, int n_algo)
{
  size_t max_ws_size = 0;
  size_t max_block_size = 0;
  size_t tmp_bytes = 0;  // Only used for filling pointer parameters that aren't used later

  int device;
  THCudaCheck(cudaGetDevice(&device));
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &tmp_bytes, &max_block_size);

  for (int i = 0; i < n_algo; i++) {
    cudnnStatus_t err;
    size_t sz;
    err = getWorkspaceSize(args, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > max_block_size)
      continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

template<typename perf_t>
perf_t getValidAlgorithm(perf_t *perfResults, const ConvolutionArgs& args, int n_algo) {

// See Note [blacklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
  bool blacklist = std::is_same<decltype(perfResults[0].algo), cudnnConvolutionBwdDataAlgo_t>::value;
  int stride_dim = args.input.dim() - 2;
  blacklist &= std::any_of(std::begin(args.params.stride),
                            std::begin(args.params.stride) + stride_dim,
                            [=](int n){return n != 1;});
#endif

  for (int i = 0; i < n_algo; i++) {
    perf_t perf = perfResults[i];

    // TODO: Shouldn't all returned results be successful?
    // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
    if (perf.status == CUDNN_STATUS_SUCCESS) {
      if (!args.params.deterministic || perf.determinism == CUDNN_DETERMINISTIC) {

        // See Note [blacklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
        bool skip = blacklist;
        skip &= (static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                  static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
        if (skip) {
          continue;
        }
#endif

        return perf;
      }
    }
  }
  AT_ERROR("no valid convolution algorithms available in CuDNN");
}

template<>
struct algorithm_search<cudnnConvolutionFwdAlgo_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<algo_t>& cache() { return fwd_algos; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    static const algo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution forward algorithms");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
    Workspace ws(max_ws_size);
    AT_CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        args.handle,
        args.idesc.desc(), args.input.data_ptr(),
        args.wdesc.desc(), args.weight.data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), args.output.data_ptr(),
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getValidAlgorithm<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getAlgorithm(
    const ConvolutionArgs& args,
    algo_t* algo)
  {
    constexpr int nalgo = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int valid_algos;
    perf_t algos[nalgo];
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        nalgo,
        &valid_algos,
        algos));
    *algo = getValidAlgorithm<perf_t>(algos, args, valid_algos).algo;
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdDataAlgo_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static BenchmarkCache<algo_t>& cache() { return bwd_data_algos; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
    Workspace ws(max_ws_size);
    AT_CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        args.handle,
        args.wdesc.desc(), args.weight.data_ptr(),
        args.odesc.desc(), args.output.data_ptr(),
        args.cdesc.desc(),
        args.idesc.desc(), args.input.data_ptr(),
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getValidAlgorithm<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getAlgorithm(const ConvolutionArgs& args, algo_t* algo) {
    constexpr int nalgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    int valid_algos;
    perf_t algos[nalgo];
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        nalgo,
        &valid_algos,
        algos));
    *algo = getValidAlgorithm<perf_t>(algos, args, valid_algos).algo;
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdFilterAlgo_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static BenchmarkCache<algo_t>& cache() { return bwd_filter_algos; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
#if CUDNN_VERSION >= 6000
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
#endif
    };
    // NOTE: - 1 because ALGO_WINOGRAD is not implemented
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
    int perf_count;
    Workspace ws(max_ws_size);

    AT_CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        args.handle,
        args.idesc.desc(), args.input.data_ptr(),
        args.odesc.desc(), args.output.data_ptr(),
        args.cdesc.desc(),
        args.wdesc.desc(), args.weight.data_ptr(),
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getValidAlgorithm<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getAlgorithm(const ConvolutionArgs& args, algo_t* algo) {
    constexpr int nalgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    int valid_algos;
    perf_t algos[nalgo];
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        nalgo,
        &valid_algos,
        algos));
    *algo = getValidAlgorithm<perf_t>(algos, args, valid_algos).algo;
  }

  static void getWorkspaceSize(const ConvolutionArgs& args, algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        workspaceSize));
  }
};

template<typename algo_t>
void findAlgorithm(const ConvolutionArgs& args, bool benchmark, algo_t* algo) {
  using search = algorithm_search<algo_t>;
  auto& cache = search::cache();

  if (cache.find(args.params, algo)) {
    return;
  }

  if (args.params.deterministic && !benchmark) {
    *algo = search::DEFAULT_ALGO;
    return;
  }

  if (!benchmark) {
    search::getAlgorithm(args, algo);
    return;
  }

  if (cache.find(args.params, algo)) {
    // re-check cache since another thread may have benchmarked the algorithm
    return;
  }

  auto perfResults = search::findAlgorithm(args);
  // for deterministic algo, look at all the perf results and return the best
  // deterministic algo
  if (perfResults.status == CUDNN_STATUS_SUCCESS &&
      !(args.params.deterministic && perfResults.determinism != CUDNN_DETERMINISTIC)) {
      *algo = perfResults.algo;
  } else {
      *algo = search::DEFAULT_ALGO;
  }
  cache.insert(args.params, *algo);

  // Free the cached blocks in our caching allocator. They are
  // needed here because the above benchmarking uses a huge amount of memory,
  // e.g. a few GBs.
  c10::cuda::CUDACachingAllocator::emptyCache();
}

template<typename algo_t>
Workspace chooseAlgorithm(
    const ConvolutionArgs& args,
    bool benchmark,
    algo_t* algo)
{
  findAlgorithm(args, benchmark, algo);
  using search = algorithm_search<algo_t>;
  size_t workspace_size;
  search::getWorkspaceSize(args, *algo, &workspace_size);
  try {
    return Workspace(workspace_size);
  } catch (std::runtime_error& e) {
    cudaGetLastError(); // clear OOM error

    // switch to default algorithm and record it in the cache to prevent
    // further OOM errors
    *algo = search::DEFAULT_ALGO;
    search::cache().insert(args.params, *algo);

    search::getWorkspaceSize(args, *algo, &workspace_size);
    return Workspace(workspace_size);
  }
}

// ---------------------------------------------------------------------
//
// Bias addition
//
// ---------------------------------------------------------------------

// In-place!
void cudnn_convolution_add_bias_nhwc_(CheckedFrom c, const TensorArg& output, const TensorArg& bias)
{
  //checkAllSameType(c, {output, bias});
  checkAllSameGPU(c, {output, bias});
  checkSize(c, bias, { output->size(output_channels_dim) });

  // See Note [CuDNN broadcast padding].  Handle the left padding
  // ourselves, but use TensorDescriptor's padding argument to do the rest.
  TensorDescriptor bdesc, odesc;
  bdesc.set(bias->expand({1, 1, 1, bias->size(0)}));
  odesc.set(*output);

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*bias);
  Constant one(dataType, 1);

  AT_CUDNN_CHECK(cudnnAddTensor(handle, &one, bdesc.desc(), bias->data_ptr(),
                                     &one, odesc.desc(), output->data_ptr()));
}

// NOTE [ Convolution design ]
//
// The general strategy:
//
//    - cudnn_convolution (Tensor)
//      Entry points for clients, takes bias
//
//    - cudnn_convolution_forward (TensorArg)
//      Entry point, which may be reused between regular
//      convolution and transposed convolution.  Does NOT take bias.
//
//    - raw_cudnn_convolution_forward_out (Tensor)
//      Low level function which invokes CuDNN, and takes an output
//      tensor which is directly written to (thus _out).
//
// Where does argument checking happen?  Here's the division of
// responsibility:
//  - Things that happen in at::Tensor
//    - TensorArg allocation
//  - Things that happen in TensorArg
//    - Check arguments (type, GPU, shape)
//
// TODO: Consider renaming zero-indexed arguments to "self"



// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes CuDNN and does not emulate support
// for group convolution on old versions of CuDNN.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//
void raw_cudnn_convolution_forward_out_nhwc(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{ input, output, weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(weight);
  args.odesc.set(output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  cudnnConvolutionFwdAlgo_t fwdAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionForward(
    args.handle,
    &one, args.idesc.desc(), input.data_ptr(),
    args.wdesc.desc(), weight.data_ptr(),
    args.cdesc.desc(), fwdAlg, workspace.data, workspace.size,
    &zero, args.odesc.desc(), output.data_ptr()));
}

Tensor cudnn_convolution_forward_nhwc(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto output_t = at::empty(conv_output_size_nhwc(input->sizes(), weight->sizes(),
                                                  padding, stride, dilation, groups),
                            input->options());
//  auto output_t = at::empty(conv_output_size_nhwc(input->sizes(), weight->sizes(),
//                                                  padding, stride, dilation, groups),
//                            torch::CUDA(at::kFloat));

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };
  // convolution_shape_check_nhwc(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous();

  raw_cudnn_convolution_forward_out_nhwc(
      *output, *input, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;
}

Tensor cudnn_convolution_nhwc(
    const Tensor& input_t, const Tensor& weight_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  CheckedFrom c = "cudnn_convolution_nhwc";
  auto output_t = cudnn_convolution_forward_nhwc(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  return output_t;
}

Tensor cudnn_convolution_with_bias_nhwc(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "cudnn_convolution_nhwc";
  auto output_t = cudnn_convolution_forward_nhwc(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);

  cudnn_convolution_add_bias_nhwc_(c, { output_t, "result", 0 }, bias);

  return output_t;
}

// NB: output_padding not needed here, as there is no ambiguity to
// resolve
Tensor cudnn_convolution_transpose_backward_input_nhwc(
    const Tensor& grad_output_t, const Tensor& weight_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  return cudnn_convolution_forward_nhwc(
    "cudnn_convolution_transpose_backward_input",
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

//std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_convolution_transpose_backward_nhwc(
//    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
//    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
//    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
//
//  Tensor grad_output = grad_output_t.contiguous();
//
//  Tensor grad_input, grad_weight, grad_bias;
//  if (output_mask[0]) {
//    grad_input = cudnn_convolution_transpose_backward_input_nhwc(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
//  }
//  if (output_mask[1]) {
//    grad_weight = cudnn_convolution_transpose_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
//  }
//  if (output_mask[2]) {
//    grad_bias = cudnn_convolution_backward_bias_nhwc(grad_output);
//  }
//
//  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
//}
//


// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_input_out_nhwc(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getCudnnDataType(grad_output);

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, grad_input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(grad_input);
  args.wdesc.set(weight);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  cudnnConvolutionBwdDataAlgo_t bwdDataAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardData(
      args.handle,
      &one, args.wdesc.desc(), weight.data_ptr(),
      args.odesc.desc(), grad_output.data_ptr(),
      args.cdesc.desc(), bwdDataAlg, workspace.data, workspace.size,
      &zero, args.idesc.desc(), grad_input.data_ptr()));
}

// NOTE [ Backward vs transpose convolutions ]
//
// Backward and transpose are algorithmically equivalent, but they
// compute their geometry differently.  In a backwards, you knew what
// the original size of the input tensor was, so you can cache that
// geometry and fill it directly.  In transposed convolution, it is
// more conventional to not explicitly specify the output (previously
// input) size, and compute it.  This, however, leaves a degree of
// freedom; this degree of freedom is resolved using the
// output_padding parameter.  Both of these interfaces are equivalent,
// but they are differently convenient depending on the use case.

Tensor cudnn_convolution_backward_input_nhwc(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  // checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  auto grad_input_t = at::empty(input_size, grad_output->options());
//  auto grad_input_t = at::empty(input_size, torch::CUDA(at::kFloat));
  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  // convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous();

  raw_cudnn_convolution_backward_input_out_nhwc(
      *grad_input, *grad_output, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *grad_input;
}

Tensor cudnn_convolution_transpose_forward_nhwc(
    CheckedFrom c,
    const TensorArg& grad_output, const TensorArg& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return cudnn_convolution_backward_input_nhwc(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_backward_input_nhwc(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  return cudnn_convolution_backward_input_nhwc(
      "cudnn_convolution_backward_input_nhwc",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_transpose_nhwc(
    const Tensor& input_t, const Tensor& weight_t,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  CheckedFrom c = "cudnn_convolution_transpose";
  auto output_t = cudnn_convolution_transpose_forward_nhwc(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  return output_t;
}

Tensor cudnn_convolution_transpose_with_bias_nhwc(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "cudnn_convolution_transpose";
  auto output_t = cudnn_convolution_transpose_forward_nhwc(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    cudnn_convolution_add_bias_nhwc_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------



void raw_cudnn_convolution_backward_weight_out_nhwc(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, grad_weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(grad_weight);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      args.handle,
      &one, args.idesc.desc(), input.data_ptr(),
      args.odesc.desc(), grad_output.data_ptr(),
      args.cdesc.desc(), bwdFilterAlg, workspace.data, workspace.size,
      &zero, args.wdesc.desc(), grad_weight.data_ptr()));
}

Tensor cudnn_convolution_backward_weight_nhwc(
    CheckedFrom c,
    IntArrayRef weight_size, const TensorArg& grad_output, const TensorArg& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{

  // checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(weight_size, grad_output->options());
//  auto grad_weight_t = torch::zeros(weight_size, torch::CUDA(at::kFloat));
  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };
  // convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  raw_cudnn_convolution_backward_weight_out_nhwc(
      *grad_weight, *grad_output, *input,
      padding, stride, dilation, groups, benchmark, deterministic);

  return grad_weight_t;
}

Tensor cudnn_convolution_backward_weight_nhwc(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  return cudnn_convolution_backward_weight_nhwc(
      "cudnn_convolution_backward_weight_nhwc",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_transpose_backward_weight_nhwc(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  return cudnn_convolution_backward_weight_nhwc(
      "cudnn_convolution_backward_weight_nhwc",
      weight_size, input, grad_output,
      padding, stride, dilation, groups, benchmark, deterministic);
}

// This is the main bprop entry-point
std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input = at::empty({}, grad_output_t.options()), grad_weight;
  if (output_mask[0]) {
    grad_input = cudnn_convolution_backward_input_nhwc(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = cudnn_convolution_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }

  return std::tuple<at::Tensor,at::Tensor>{grad_input, grad_weight};
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = cudnn_convolution_transpose_backward_input_nhwc(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = cudnn_convolution_transpose_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }

  return std::tuple<Tensor,Tensor>{grad_input, grad_weight};
}

// ---------------------------------------------------------------------
//
// Convolution backward (bias)
//
// ---------------------------------------------------------------------

Tensor cudnn_convolution_backward_bias_nhwc(
    const Tensor& grad_output_t)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 };

  auto grad_bias_t = at::empty({ grad_output->size(output_channels_dim) }, grad_output->options());
//  auto grad_bias_t = torch::zeros({ grad_output->size(output_channels_dim) }, torch::CUDA(at::kFloat));

  TensorArg grad_bias{ grad_bias_t, "result", 0 };

  // See Note [CuDNN broadcast padding].  Handle the left padding
  // ourselves, but use TensorDescriptor's pad argument to do the rest.
  TensorDescriptor bdesc{grad_bias->expand({1, 1, 1, grad_bias->size(0)})};
  TensorDescriptor odesc{*grad_output};

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*grad_bias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &one, odesc.desc(), grad_output->data_ptr(),
                                                   &zero, bdesc.desc(), grad_bias->data_ptr()));
  return *grad_bias;
}

// This is the main bprop entry-point
std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_convolution_backward_with_bias_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input = at::empty({}, grad_output_t.options()), grad_weight, grad_bias;
//  Tensor grad_input = torch::zeros({}, torch::CUDA(at::kFloat)), grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = cudnn_convolution_backward_input_nhwc(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = cudnn_convolution_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }

  grad_bias = cudnn_convolution_backward_bias_nhwc(grad_output);

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}


std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_convolution_transpose_backward_with_bias_nhwc(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    std::vector<long> padding, std::vector<long> output_padding, std::vector<long> stride, std::vector<long> dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = cudnn_convolution_transpose_backward_input_nhwc(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = cudnn_convolution_transpose_backward_weight_nhwc(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[2]) {
    grad_bias = cudnn_convolution_backward_bias_nhwc(grad_output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

}}}  // namespace
