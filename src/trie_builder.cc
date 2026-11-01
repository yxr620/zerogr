#include "trie_builder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace zerogr {
namespace {

template <class Token> HostTrie<Token> BuildRows(
    const Token *sids, size_t count, size_t length, size_t stride) {
  constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();
  constexpr size_t kMaxEdges = std::numeric_limits<uint32_t>::max() - size_t(1);
  if (length == 0 || (count && !sids))
    throw std::invalid_argument("SID length must be positive; nonempty input needs data");
  if (length > kMaxSize / sizeof(Token))
    throw std::length_error("SID row byte size overflow");
  const size_t row_bytes = length * sizeof(Token);
  if (stride == 0) stride = row_bytes;
  if (stride < row_bytes || stride % alignof(Token) != 0 ||
      (sids && reinterpret_cast<uintptr_t>(sids) % alignof(Token) != 0))
    throw std::invalid_argument("invalid SID row stride or alignment");
  if (count && (count - 1 > (kMaxSize - row_bytes) / stride ||
      reinterpret_cast<uintptr_t>(sids) >
          std::numeric_limits<uintptr_t>::max() - ((count - 1) * stride + row_bytes)))
    throw std::length_error("SID input address range overflow");
  auto Row = [&](size_t index) {
    return reinterpret_cast<const Token *>(
        reinterpret_cast<const unsigned char *>(sids) + index * stride);
  };
  for (size_t i = 0; i < count; ++i) {
    const Token *row = Row(i);
    for (size_t d = 0; d < length; ++d)
      if (uint64_t(row[d]) > uint64_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("token exceeds decoder int32_t output range");
    if (i && !std::lexicographical_compare(Row(i - 1), Row(i - 1) + length,
                                           row, row + length))
      throw std::invalid_argument("SID rows must be sorted and unique");
  }
  auto PrefixEqual = [&](size_t a, size_t b, size_t depth) {
    return std::equal(Row(a), Row(a) + depth, Row(b));
  };
  // Two-pass, breadth-first construction with implicit child IDs.
  size_t edge_count = 0;
  for (size_t depth = 0; depth < length && count; ++depth) {
    for (size_t begin = 0; begin < count;) {
      size_t end = begin + 1;
      while (end < count && PrefixEqual(begin, end, depth + 1)) ++end;
      if (edge_count == kMaxEdges) throw std::length_error("CSR edge capacity exceeded");
      ++edge_count;
      begin = end;
    }
  }
  HostTrie<Token> trie;
  trie.col.reserve(edge_count);
  trie.row_ptr.reserve(edge_count + 2);
  trie.row_ptr.push_back(0);
  trie.stored_states = 1;
  if (!count) {
    trie.row_ptr.push_back(0);
    return trie;
  }
  std::vector<uint32_t> state_ids(count, 0);
  for (size_t depth = 0; depth < length; ++depth) {
    uint32_t open_parent = std::numeric_limits<uint32_t>::max();
    for (size_t begin = 0; begin < count;) {
      size_t end = begin + 1;
      while (end < count && PrefixEqual(begin, end, depth + 1)) ++end;
      const uint32_t parent = state_ids[begin];
      if (open_parent != std::numeric_limits<uint32_t>::max() && parent != open_parent)
        trie.row_ptr.push_back(static_cast<uint32_t>(trie.col.size()));
      open_parent = parent;
      const uint32_t child = static_cast<uint32_t>(trie.col.size() + 1);
      trie.col.push_back(Row(begin)[depth]);
      if (depth + 1 < length) {
        ++trie.stored_states;
        for (size_t row = begin; row < end; ++row) state_ids[row] = child;
      }
      begin = end;
    }
    trie.row_ptr.push_back(static_cast<uint32_t>(trie.col.size()));
  }
  trie.edges = static_cast<uint32_t>(trie.col.size());
  if (trie.row_ptr.size() != size_t(trie.stored_states) + 1)
    throw std::logic_error("leaf-trimmed CSR row count mismatch");
  for (size_t state = 0; state < trie.stored_states; ++state)
    trie.max_degree = std::max(trie.max_degree, trie.row_ptr[state + 1] - trie.row_ptr[state]);
  return trie;
}

}  // namespace

HostTrie<uint16_t> BuildTrieFromSortedUnique(
    const uint16_t *sids, size_t count, size_t length, size_t stride) {
  return BuildRows(sids, count, length, stride);
}

}  // namespace zerogr
