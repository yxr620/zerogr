#pragma once
#include <cstdint>

namespace zerogr {
namespace detail {
inline constexpr int kPaddedCandidatesPerTask = 64;
struct PaddedProjectionTask {
  uint32_t beam;
  uint32_t candidate_begin;
  uint16_t candidate_count;
  uint16_t reserved;
};
static_assert(sizeof(PaddedProjectionTask) == 12);
}  // namespace detail
}  // namespace zerogr
