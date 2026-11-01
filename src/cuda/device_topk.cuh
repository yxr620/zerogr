#pragma once

#include <cub/device/device_topk.cuh>
#include <cuda/stream>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace zerogr::detail {

__global__ void InitIndices(uint32_t *indices, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) indices[i] = static_cast<uint32_t>(i);
}

// DeviceTopK returns unsorted pairs; the next step consumes the same beam order.
__global__ void GatherTopK(const float *candidate_scores, const uint16_t *candidate_tokens,
    const uint32_t *candidate_states, const uint32_t *indices, int selected, int beam,
    float *scores, uint16_t *tokens, uint32_t *states) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= beam) return;
  const uint32_t index = i < selected ? indices[i] : 0;
  const bool valid = i < selected && isfinite(candidate_scores[index]);
  scores[i] = valid ? candidate_scores[index] : -__int_as_float(0x7f800000);
  tokens[i] = valid ? candidate_tokens[index] : 0;
  states[i] = valid ? candidate_states[index] : 0;
}

inline cudaError_t DeviceTopK(void *storage, size_t &bytes, const float *scores,
    float *selected_scores, const uint32_t *indices, uint32_t *selected_indices,
    int count, int k) {
  cuda::stream_ref stream{cudaStream_t{}};
  auto requirements = cuda::execution::require(
      cuda::execution::determinism::not_guaranteed,
      cuda::execution::output_ordering::unsorted);
  auto env = cuda::std::execution::env{stream, requirements};
  return cub::DeviceTopK::MaxPairs(storage, bytes, scores, selected_scores,
      indices, selected_indices, count, k, env);
}

}  // namespace zerogr::detail
