#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace zerogr {
namespace detail {

// Warp reduction for the grouped selective dot products.
__device__ inline float WarpReduceSum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ inline float DotFloat4WithScalarTail(const float *hidden,
                                         const float *codebook,
                                         int hidden_dim, int lane) {
  float sum = 0.0f;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(hidden) & 0xfu) == 0 &&
      (reinterpret_cast<uintptr_t>(codebook) & 0xfu) == 0;
  if (aligned) {
    const int vector_count = hidden_dim / 4;
    const float4 *hidden4 = reinterpret_cast<const float4 *>(hidden);
    const float4 *codebook4 = reinterpret_cast<const float4 *>(codebook);
    for (int index = lane; index < vector_count; index += warpSize) {
      const float4 lhs = hidden4[index];
      const float4 rhs = codebook4[index];
      sum += lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
    }
    for (int dim = vector_count * 4 + lane; dim < hidden_dim;
         dim += warpSize) {
      sum += hidden[dim] * codebook[dim];
    }
  } else {
    for (int dim = lane; dim < hidden_dim; dim += warpSize) {
      sum += hidden[dim] * codebook[dim];
    }
  }
  return sum;
}

// Full projection uses row-major [beam, token] storage.
__device__ inline float GatherCandidateLogit(
    const float *logits, int row, int vocab, int token) {
  return logits[static_cast<size_t>(row) * vocab + token];
}

}  // namespace detail
}  // namespace zerogr
