#include "options.h"
#include "synthetic_data.h"
#include "zerogr/decoder.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

float Median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

int Run(int argc, char **argv) {
  const auto options = zerogr::example::ParseArgs(argc, argv);
  const auto workload = zerogr::example::MakeSyntheticWorkload(options);
  std::cout << "ZeroGR: sampled=" << options.samples
      << " unique=" << workload.unique_rows
      << " stored_states=" << workload.trie.stored_states
      << " edges=" << workload.trie.edges << '\n'
      << std::fixed << std::setprecision(3);

  zerogr::Decoder decoder(options.decode, workload.trie, workload.hidden, workload.weights);
  decoder.Calibrate(options.calibration);
  const auto &plan = decoder.plan();
  for (size_t d = 0; d < plan.size(); ++d) {
    const auto &p = plan[d];
    std::cout << "depth=" << d << " candidates=" << p.candidates
        << " projection=" << (p.projection == zerogr::ProjectionBackend::kDense ? "dense" : "selective")
        << " dense_us=" << p.dense_us << " selective_us=" << p.selective_us
        << " topk=" << (p.topk == zerogr::TopKBackend::kBlock ? "block" : "device") << '\n';
  }
  for (int i = 0; i < options.calibration.warmup; ++i) decoder.Decode();
  std::vector<float> times;
  for (int i = 0; i < options.calibration.trials; ++i) times.push_back(decoder.Decode());
  std::cout << "ZeroGR median_decode_us=" << Median(times) << '\n';

  const auto result = decoder.Download();
  const auto best = std::max_element(result.scores.begin(), result.scores.end());
  const auto finite = std::count_if(result.scores.begin(), result.scores.end(),
      [](float x) { return std::isfinite(x); });
  std::cout << "finite_beams=" << finite << " best_score=" << *best
      << " last_token=" << result.tokens[size_t(best - result.scores.begin())] << '\n';
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try { return Run(argc, argv); }
  catch (const std::exception &error) {
    std::cerr << "ZeroGR: " << error.what() << '\n';
    return 1;
  }
}
