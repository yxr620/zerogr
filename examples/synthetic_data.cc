#include "synthetic_data.h"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace zerogr::example {
namespace {

size_t Product(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::length_error("synthetic input size overflow");
  return a * b;
}

}  // namespace

SyntheticWorkload MakeSyntheticWorkload(const Options &options) {
  options.decode.Validate();
  if (options.samples < 1 || options.seed < 0)
    throw std::invalid_argument("invalid synthetic input configuration");
  const auto &config = options.decode;
  std::mt19937_64 rng(options.seed);
  std::uniform_int_distribution<int> token(0, config.vocab_size - 1);
  std::vector<std::vector<uint16_t>> rows(options.samples,
      std::vector<uint16_t>(config.sid_length));
  for (auto &row : rows)
    for (auto &t : row) t = static_cast<uint16_t>(token(rng));
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

  SyntheticWorkload workload;
  workload.unique_rows = rows.size();
  std::vector<uint16_t> flat;
  flat.reserve(Product(rows.size(), size_t(config.sid_length)));
  for (const auto &row : rows) flat.insert(flat.end(), row.begin(), row.end());
  workload.trie = BuildTrieFromSortedUnique(flat.data(), rows.size(), config.sid_length);

  rng.seed(uint64_t(options.seed) + 1);
  std::uniform_real_distribution<float> value(-0.1f, 0.1f);
  workload.hidden.resize(Product(Product(size_t(config.sid_length), size_t(config.beam_width)),
      size_t(config.hidden_dim)));
  workload.weights.resize(Product(size_t(config.vocab_size), size_t(config.hidden_dim)));
  // The two TopK backends may emit beams in different orders.
  for (int d = 0; d < config.sid_length; ++d) {
    const size_t begin = size_t(d) * config.beam_width * config.hidden_dim;
    for (int h = 0; h < config.hidden_dim; ++h) workload.hidden[begin + h] = value(rng);
    for (int b = 1; b < config.beam_width; ++b)
      std::copy_n(workload.hidden.data() + begin, config.hidden_dim,
          workload.hidden.data() + begin + size_t(b) * config.hidden_dim);
  }
  for (auto &x : workload.weights) x = value(rng);
  return workload;
}

}  // namespace zerogr::example
