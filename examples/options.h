#pragma once

#include "zerogr/config.h"

namespace zerogr::example {

struct Options {
  DecodeConfig decode;
  CalibrationConfig calibration;
  int samples = 10000;
  int seed = 1234;
};

Options ParseArgs(int argc, char **argv);

}  // namespace zerogr::example
