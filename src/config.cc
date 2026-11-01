#include "zerogr/config.h"

#include <stdexcept>

namespace zerogr {

void DecodeConfig::Validate() const {
  if (vocab_size < 1 || vocab_size > 65536 || sid_length < 1 || hidden_dim < 1 ||
      beam_width < 64 || beam_width > 512)
    throw std::invalid_argument("invalid decode configuration");
}

void CalibrationConfig::Validate() const {
  if (warmup < 0 || trials < 1)
    throw std::invalid_argument("invalid calibration configuration");
}

}  // namespace zerogr
