#pragma once

#include "options.h"
#include "zerogr/trie.h"

#include <cstddef>
#include <vector>

namespace zerogr::example {

struct SyntheticWorkload {
  HostTrie<uint16_t> trie;
  size_t unique_rows = 0;
  std::vector<float> hidden;
  std::vector<float> weights;
};

SyntheticWorkload MakeSyntheticWorkload(const Options &options);

}  // namespace zerogr::example
