#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zerogr {

// CPU-only value-free CSR. Root is 0; child state is edge_index + 1.
// Only depths [0, sid_length) have rows; complete SID leaves are virtual.
// Empty input retains one empty root row: row_ptr={0,0}, stored_states=1.
template <class Token> struct HostTrie {
  std::vector<uint32_t> row_ptr;
  std::vector<Token> col;
  uint32_t stored_states = 0;
  uint32_t edges = 0;
  uint32_t max_degree = 0;
};

// Input consists of sid_count strictly lexicographically sorted, unique rows,
// each containing exactly sid_length tokens. Sort/deduplicate before calling.
// row_stride_bytes=0 means contiguous rows; a larger stride supports array or
// struct records without copying SID data. No PID payloads are read or stored.
// Tokens are uint16_t.
// sid_length must be positive, even for empty input. A null pointer is allowed
// only for sid_count=0. Invalid shape/order/duplicates/tokens throw
// std::invalid_argument; size/address/CSR overflow throws std::length_error.
// Allocation failures propagate std::bad_alloc. The builder performs no CUDA
// operations; upload and workspace allocation belong outside decode timing.
HostTrie<uint16_t> BuildTrieFromSortedUnique(
    const uint16_t *sids, size_t sid_count, size_t sid_length,
    size_t row_stride_bytes = 0);

}  // namespace zerogr
