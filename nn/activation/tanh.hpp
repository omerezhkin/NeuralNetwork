#pragma once

#include "core/math/Linalg.hpp"

#include <cmath>

namespace nn::activation {

struct Tanh {
  [[nodiscard]] static nn::Scalar forward(nn::Scalar x) { return std::tanh(x); }

  [[nodiscard]] static nn::Scalar derivative_from_output(nn::Scalar y) {
    return 1.F - y * y;
  }
};

}  // namespace nn::activation
