#pragma once

#include "zerogr/config.h"
#include "zerogr/trie.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace zerogr {

enum class ProjectionBackend { kDense, kSelective };
enum class TopKBackend { kDevice, kBlock };

struct DepthPlan {
  ProjectionBackend projection = ProjectionBackend::kSelective;
  TopKBackend topk = TopKBackend::kDevice;
  uint64_t candidates = 0;
  float dense_us = 0;
  float selective_us = 0;
};

// Final beam entries. Token is the last emitted token; padding has score=-inf.
struct DecodeResult {
  std::vector<float> scores;
  std::vector<uint16_t> tokens;
  std::vector<uint32_t> states;
};

class Decoder {
 public:
  // Copies the trie, FP32 hidden[length, beam, dim] and weights[vocab, dim]
  // to the current CUDA device. Inputs remain fixed for this decoder's lifetime.
  // Hidden vectors must be identical across beams within each depth so the
  // synthetic workload is independent of the two TopK backends' output order.
  Decoder(const DecodeConfig &config, const HostTrie<uint16_t> &trie,
          const std::vector<float> &hidden, const std::vector<float> &weights);
  ~Decoder();
  Decoder(const Decoder &) = delete;
  Decoder &operator=(const Decoder &) = delete;

  // Selects projection and TopK backends per depth for the fixed inputs.
  void Calibrate(const CalibrationConfig &config = {});
  const std::vector<DepthPlan> &plan() const;

  // Runs a complete decode after calibration and waits for completion.
  // Returns CUDA event time in microseconds, excluding result copies/checks.
  float Decode();
  DecodeResult Download() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace zerogr
