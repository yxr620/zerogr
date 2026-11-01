#pragma once

namespace zerogr {

struct DecodeConfig {
  int vocab_size = 256;
  int sid_length = 5;
  int hidden_dim = 256;
  int beam_width = 128;

  void Validate() const;
};

struct CalibrationConfig {
  int warmup = 2;
  int trials = 5;

  void Validate() const;
};

}  // namespace zerogr
