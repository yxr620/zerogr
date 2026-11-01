#pragma once

#include "projection_primitives.cuh"
#include "projection_tasks.h"

namespace zerogr {
namespace detail {
constexpr int kThreads = 256;
enum PaddedProjectionError : uint32_t {
  kCandidateBoundsError = 1u << 0,
  kTaskBoundsError = 1u << 1,
  kTokenOrStateBoundsError = 1u << 3,
};

__device__ inline void RecordError(uint32_t *error, PaddedProjectionError value) {
  atomicOr(error, static_cast<uint32_t>(value));
}

__host__ __device__ constexpr uint32_t TaskCountForDegree(uint32_t degree) {
  const uint32_t candidates_per_task =
      static_cast<uint32_t>(kPaddedCandidatesPerTask);
  return degree / candidates_per_task +
         (degree % candidates_per_task != 0 ? 1u : 0u);
}

static_assert(TaskCountForDegree(0) == 0);
static_assert(TaskCountForDegree(1) == 1);
static_assert(TaskCountForDegree(64) == 1);
static_assert(TaskCountForDegree(65) == 2);

// Expands each beam's trie row into padded candidate slots and emits work-queue
// tasks covering only its valid candidates.
template <class Token> __global__ void PaddedLookupAndBuildTasksKernel(
    const uint32_t *states, const float *parent_scores,
    const uint32_t *row_ptr, uint32_t stored_state_count, const Token *col,
    int active_beams, int limit,
    int vocab_size,
    uint32_t candidate_capacity, // active_beams * limit
    Token *candidate_tokens,
    uint32_t *candidate_states, uint8_t *candidate_valid,
    float *candidate_scores, PaddedProjectionTask *tasks,
    uint32_t *task_count, uint32_t task_capacity, uint64_t *actual_count,
    uint32_t *error) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (static_cast<uint32_t>(index) >= candidate_capacity) {
    return;
  }

  const int beam = index / limit;
  const int offset = index - beam * limit;
  const bool parent_valid = isfinite(parent_scores[beam]);
  const uint32_t state = states[beam];
  bool state_valid = state < stored_state_count;
  uint32_t begin = 0;
  uint32_t degree = 0;
  if (state_valid) {
    begin = row_ptr[state];
    const uint32_t end = row_ptr[state + 1];
    state_valid = end >= begin;
    degree = state_valid ? end - begin : 0;
  }
  if (!state_valid) {
    if (offset == 0) {
      RecordError(error, kTokenOrStateBoundsError);
    }
    degree = 0;
  }

  const bool valid = parent_valid &&
                     static_cast<uint32_t>(offset) < degree;
  Token token = 0;
  uint32_t next_state = 0;
  if (valid) {
    const uint32_t edge = begin + static_cast<uint32_t>(offset);
    token = col[edge];
    if (static_cast<uint32_t>(token) >= static_cast<uint32_t>(vocab_size)) {
      RecordError(error, kTokenOrStateBoundsError);
    } else {
      next_state = edge + 1;
    }
  }
  candidate_tokens[index] = token;
  candidate_states[index] = next_state;
  candidate_valid[index] = valid ? 1u : 0u;
  candidate_scores[index] = -__int_as_float(0x7f800000);

  // The first thread of each beam builds its tasks.
  if (offset != 0 || !parent_valid || !state_valid || degree == 0) {
    return;
  }
  if (static_cast<uint64_t>(beam) * static_cast<uint64_t>(limit) + degree >
      candidate_capacity) {
    RecordError(error, kCandidateBoundsError);
    return;
  }
  if (actual_count != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long *>(actual_count),
              static_cast<unsigned long long>(degree));
  }

  const uint32_t beam_task_count = TaskCountForDegree(degree);
  const uint32_t task_begin = atomicAdd(task_count, beam_task_count);
  if (static_cast<uint64_t>(task_begin) + beam_task_count > task_capacity) {
    RecordError(error, kTaskBoundsError);
    return;
  }
  const uint32_t candidate_begin =
      static_cast<uint32_t>(beam) * static_cast<uint32_t>(limit);
  for (uint32_t local_task = 0; local_task < beam_task_count; ++local_task) {
    const uint32_t local_begin =
        local_task * static_cast<uint32_t>(kPaddedCandidatesPerTask);
    const uint32_t count = min(
        degree - local_begin,
        static_cast<uint32_t>(kPaddedCandidatesPerTask));
    tasks[task_begin + local_task] = {
        static_cast<uint32_t>(beam), candidate_begin + local_begin,
        static_cast<uint16_t>(count), 0};
  }
}

__device__ inline bool TaskIsValid(const PaddedProjectionTask &task,
                            int active_beams,
                            uint32_t candidate_capacity) {
  const uint64_t candidate_end =
      static_cast<uint64_t>(task.candidate_begin) + task.candidate_count;
  return task.beam < static_cast<uint32_t>(active_beams) &&
         task.candidate_count > 0 &&
         task.candidate_count <= kPaddedCandidatesPerTask &&
         candidate_end <= candidate_capacity;
}

// Consumes padded work-queue tasks, caching one beam's hidden vector per block
// while warps score the task's candidate tokens.
template <class Token> __global__ void PaddedGroupedSelectiveProjectScoreKernel(
    const float *hidden, const float *codebook, const float *parent_scores,
    const Token *candidate_tokens, const PaddedProjectionTask *tasks,
    const uint32_t *task_count, uint32_t task_capacity, int hidden_dim,
    int vocab_size, int active_beams, uint32_t candidate_capacity,
    float *candidate_scores, uint32_t *error) {
  extern __shared__ float shared_hidden[];

  const uint32_t actual_task_count = *task_count;
  if (actual_task_count > task_capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      RecordError(error, kTaskBoundsError);
    }
    return;
  }

  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  constexpr int kWarpsPerBlock = kThreads / 32;
  for (uint32_t task_index = blockIdx.x; task_index < actual_task_count;
       task_index += gridDim.x) {
    const PaddedProjectionTask task = tasks[task_index];
    const bool valid_task =
        TaskIsValid(task, active_beams, candidate_capacity);
    if (!valid_task && threadIdx.x == 0) {
      RecordError(error, kTaskBoundsError);
    }
    if (valid_task) {
      for (int dim = threadIdx.x; dim < hidden_dim; dim += blockDim.x) {
        shared_hidden[dim] =
            hidden[static_cast<size_t>(task.beam) * hidden_dim + dim];
      }
    }
    __syncthreads();

    if (valid_task) {
      for (int wave = 0; wave < task.candidate_count;
           wave += kWarpsPerBlock) {
        const int local_candidate = wave + warp;
        if (local_candidate < task.candidate_count) {
          const uint32_t candidate =
              task.candidate_begin + static_cast<uint32_t>(local_candidate);
          const uint32_t token = candidate_tokens[candidate];
          if (token < static_cast<uint32_t>(vocab_size)) {
            float sum = DotFloat4WithScalarTail(
                shared_hidden,
                codebook + static_cast<size_t>(token) * hidden_dim,
                hidden_dim, lane);
            sum = WarpReduceSum(sum);
            if (lane == 0) {
              candidate_scores[candidate] = parent_scores[task.beam] + sum;
            }
          } else if (lane == 0) {
            RecordError(error, kTokenOrStateBoundsError);
          }
        }
      }
    }
    __syncthreads();
  }
}

// Expands each active beam's trie row into fixed-width padded candidate slots.
template <class Token> __global__ void LookupCandidatesKernel(
    const uint32_t *states, const float *parent_scores,
    const uint32_t *row_ptr, const Token *col,
    int active_beams, int limit, Token *candidate_tokens,
    uint32_t *candidate_states, uint8_t *candidate_valid,
    uint64_t *actual_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = active_beams * limit;
  if (idx >= total) {
    return;
  }

  const int beam = idx / limit;
  const int offset = idx - beam * limit;
  const uint32_t state = states[beam];
  const uint32_t begin = row_ptr[state];
  const uint32_t end = row_ptr[state + 1];
  const bool valid = isfinite(parent_scores[beam]) &&
                     static_cast<uint32_t>(offset) < end - begin;

  Token token = 0;
  uint32_t next_state = 0;
  if (valid) {
    const uint32_t edge = begin + static_cast<uint32_t>(offset);
    token = col[edge];
    next_state = edge + 1;
    if (actual_count != nullptr) {
      atomicAdd(reinterpret_cast<unsigned long long *>(actual_count), 1ULL);
    }
  }
  candidate_tokens[idx] = token;
  candidate_states[idx] = next_state;
  candidate_valid[idx] = valid ? 1 : 0;
}

// Gathers full-projection logits for valid padded candidates and masks padding.
template <class Token> __global__ void GatherDenseScoreKernel(
    const float *dense_logits, const float *parent_scores,
    const Token *candidate_tokens, const uint8_t *candidate_valid, int limit,
    int candidate_count, int vocab_size, float *candidate_scores) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= candidate_count) {
    return;
  }
  if (candidate_valid[idx] == 0) {
    candidate_scores[idx] = -__int_as_float(0x7f800000);
    return;
  }
  const int beam = idx / limit;
  const int token = static_cast<int>(candidate_tokens[idx]);
  candidate_scores[idx] =
      parent_scores[beam] + GatherCandidateLogit(dense_logits, beam, vocab_size, token);
}

}  // namespace detail
}  // namespace zerogr
