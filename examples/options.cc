#include "options.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace zerogr::example {
namespace {

void Usage() {
  std::cout << "ZeroGR: synthetic constrained decoding\n"
    "  --num-samples N  sampled SID rows (default 10000)\n"
    "  --vocab-size N   1..65536 (default 256)\n"
    "  --sid-length N   tokens per SID (default 5)\n"
    "  --hidden-dim N   FP32 projection dimension (default 256)\n"
    "  --beam-width N   64..512; block TopK uses 128 (default 128)\n"
    "  --seed N         nonnegative seed (default 1234)\n"
    "  --warmup N       warmups per path (default 2)\n"
    "  --trials N       calibration and decode trials (default 5)\n";
}

}  // namespace

Options ParseArgs(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") { Usage(); std::exit(0); }
    int *value = nullptr;
    if (key == "--num-samples") value = &o.samples;
    else if (key == "--vocab-size") value = &o.decode.vocab_size;
    else if (key == "--sid-length") value = &o.decode.sid_length;
    else if (key == "--hidden-dim") value = &o.decode.hidden_dim;
    else if (key == "--beam-width") value = &o.decode.beam_width;
    else if (key == "--seed") value = &o.seed;
    else if (key == "--warmup") value = &o.calibration.warmup;
    else if (key == "--trials") value = &o.calibration.trials;
    else throw std::invalid_argument("unknown option: " + key);
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    const std::string text = argv[i];
    size_t consumed = 0;
    const long long n = std::stoll(text, &consumed);
    if (consumed != text.size() || n < 0 || n > std::numeric_limits<int>::max())
      throw std::invalid_argument("invalid value for " + key);
    *value = static_cast<int>(n);
  }
  if (o.samples == 0) throw std::invalid_argument("num-samples must be positive");
  o.decode.Validate();
  o.calibration.Validate();
  return o;
}

}  // namespace zerogr::example
