#pragma once
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <limits>
#include "projection_tasks.h"

namespace zerogr {
namespace detail {
inline constexpr int kSmallTopKBeamWidth = 128;
inline constexpr int kSmallTopKOutputCount = 128;
inline constexpr int kSmallTopKMaxCandidates = 512;
inline constexpr int kSmallTopKThreads = 256;
inline constexpr int kSmallTopKItemsPerThread = 2;

static_assert(kSmallTopKThreads * kSmallTopKItemsPerThread ==
              kSmallTopKMaxCandidates);
static_assert(kSmallTopKOutputCount == kSmallTopKBeamWidth);

enum SmallTopKError : uint32_t {
  kSmallTopKTaskCountBoundsError = 1u << 0,
  kSmallTopKTaskShapeBoundsError = 1u << 1,
  kSmallTopKCandidateIntervalBoundsError = 1u << 2,
  kSmallTopKCandidateCountOverflowError = 1u << 3,
  kSmallTopKInvalidScoreError = 1u << 4,
  kSmallTopKInvalidSortedCandidateIndexError = 1u << 6,
};

// Task-aware block TopK with fused output gathering.
using SmallTopKBlockScan = cub::BlockScan<uint32_t, kSmallTopKThreads>;
using SmallTopKBlockSort =
    cub::BlockRadixSort<float, kSmallTopKThreads,
                        kSmallTopKItemsPerThread, uint32_t>;

inline constexpr uint32_t kInvalidCandidateIndex =
    std::numeric_limits<uint32_t>::max();

__device__ inline float NegativeInfinity() {
  return -__int_as_float(0x7f800000);
}

__device__ inline void RecordBlockError(uint32_t *error, uint32_t value) {
  atomicOr(error, value);
}

template <class Token> __global__ void TaskAwareSmallTopKAndPostKernel(
    const float *candidate_scores, const Token *candidate_tokens,
    const uint32_t *candidate_states, const PaddedProjectionTask *tasks,
    const uint32_t *task_count, uint32_t task_capacity,
    uint32_t candidate_capacity, int active_beams, float *next_scores,
    Token *next_tokens, uint32_t *next_states, uint32_t *error) {
  __shared__ typename SmallTopKBlockScan::TempStorage scan_storage;
  __shared__ typename SmallTopKBlockSort::TempStorage sort_storage;
  __shared__ uint32_t shared_task_count;
  __shared__ uint32_t shared_candidate_count;
  __shared__ uint32_t shared_error;
  __shared__ uint32_t shared_indices[kSmallTopKMaxCandidates];

  if (threadIdx.x == 0) {
    shared_task_count = *task_count;
    shared_candidate_count = 0;
    shared_error = 0;
  }
  __syncthreads();

  const bool task_count_valid =
      shared_task_count <= task_capacity &&
      shared_task_count <= static_cast<uint32_t>(kSmallTopKMaxCandidates);
  if (threadIdx.x == 0 && !task_count_valid) {
    RecordBlockError(&shared_error, kSmallTopKTaskCountBoundsError);
  }

  uint32_t scan_input[kSmallTopKItemsPerThread] = {};
  uint32_t task_offsets[kSmallTopKItemsPerThread] = {};
  PaddedProjectionTask thread_tasks[kSmallTopKItemsPerThread] = {};
  const uint32_t safe_task_count = task_count_valid ? shared_task_count : 0;
#pragma unroll
  for (int item = 0; item < kSmallTopKItemsPerThread; ++item) {
    const uint32_t task_index =
        static_cast<uint32_t>(threadIdx.x * kSmallTopKItemsPerThread + item);
    if (task_index < safe_task_count) {
      const PaddedProjectionTask task = tasks[task_index];
      thread_tasks[item] = task;
      uint32_t task_error = 0;
      if (task.candidate_count == 0 ||
          task.candidate_count > kPaddedCandidatesPerTask ||
          task.beam >= static_cast<uint32_t>(active_beams)) {
        task_error |= kSmallTopKTaskShapeBoundsError;
      }
      const uint64_t candidate_end =
          static_cast<uint64_t>(task.candidate_begin) +
          static_cast<uint64_t>(task.candidate_count);
      if (candidate_end > candidate_capacity) {
        task_error |= kSmallTopKCandidateIntervalBoundsError;
      }
      if (task_error != 0) {
        RecordBlockError(&shared_error, task_error);
      } else {
        scan_input[item] = task.candidate_count;
      }
    }
  }

  uint32_t candidate_count = 0;
  SmallTopKBlockScan(scan_storage).ExclusiveSum(
      scan_input, task_offsets, candidate_count);
  if (threadIdx.x == 0) {
    shared_candidate_count = candidate_count;
    if (candidate_count >
        static_cast<uint32_t>(kSmallTopKMaxCandidates)) {
      RecordBlockError(&shared_error,
                       kSmallTopKCandidateCountOverflowError);
    }
  }
  __syncthreads();

  if (shared_error == 0) {
#pragma unroll
    for (int item = 0; item < kSmallTopKItemsPerThread; ++item) {
      const uint32_t task_index = static_cast<uint32_t>(
          threadIdx.x * kSmallTopKItemsPerThread + item);
      if (task_index < safe_task_count) {
        const PaddedProjectionTask task = thread_tasks[item];
        for (uint32_t local = 0; local < task.candidate_count; ++local) {
          shared_indices[task_offsets[item] + local] =
              task.candidate_begin + local;
        }
      }
    }
  }
  __syncthreads();

  float thread_scores[kSmallTopKItemsPerThread];
  uint32_t thread_indices[kSmallTopKItemsPerThread];
#pragma unroll
  for (int item = 0; item < kSmallTopKItemsPerThread; ++item) {
    const uint32_t rank = static_cast<uint32_t>(
        threadIdx.x * kSmallTopKItemsPerThread + item);
    float score = NegativeInfinity();
    uint32_t candidate = kInvalidCandidateIndex;
    if (shared_error == 0 && rank < shared_candidate_count) {
      candidate = shared_indices[rank];
      score = candidate_scores[candidate];
      if (!isfinite(score)) {
        RecordBlockError(&shared_error, kSmallTopKInvalidScoreError);
        score = NegativeInfinity();
        candidate = kInvalidCandidateIndex;
      }
    }
    thread_scores[item] = score;
    thread_indices[item] = candidate;
  }
  __syncthreads();

  SmallTopKBlockSort(sort_storage).SortDescending(thread_scores,
                                                  thread_indices);
  __syncthreads();

  // BlockRadixSort returns a blocked arrangement: thread t owns consecutive
  // ranks [t * items_per_thread, ...]. Materialize that layout explicitly so
  // output threads can gather ranks without relying on a striped convention.
#pragma unroll
  for (int item = 0; item < kSmallTopKItemsPerThread; ++item) {
    const uint32_t rank = static_cast<uint32_t>(
        threadIdx.x * kSmallTopKItemsPerThread + item);
    shared_indices[rank] = thread_indices[item];
  }
  __syncthreads();

  if (threadIdx.x < kSmallTopKOutputCount && shared_error == 0 &&
      static_cast<uint32_t>(threadIdx.x) < shared_candidate_count) {
    const uint32_t candidate = shared_indices[threadIdx.x];
    if (candidate == kInvalidCandidateIndex ||
        candidate >= candidate_capacity) {
      RecordBlockError(&shared_error,
                       kSmallTopKInvalidSortedCandidateIndexError);
    }
  }
  __syncthreads();

  if (threadIdx.x < kSmallTopKOutputCount) {
    const uint32_t rank = static_cast<uint32_t>(threadIdx.x);
    if (shared_error == 0 && rank < shared_candidate_count) {
      const uint32_t candidate = shared_indices[rank];
      next_scores[rank] = candidate_scores[candidate];
      next_tokens[rank] = candidate_tokens[candidate];
      next_states[rank] = candidate_states[candidate];
    } else {
      next_scores[rank] = NegativeInfinity();
      next_tokens[rank] = 0;
      next_states[rank] = 0;
    }
  }
  if (threadIdx.x == 0 && shared_error != 0) {
    atomicOr(error, shared_error);
  }
}

}  // namespace detail
}  // namespace zerogr
