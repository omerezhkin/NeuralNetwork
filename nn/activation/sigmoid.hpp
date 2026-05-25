#pragma once

#include "core/math/Linalg.hpp"

#include <cmath>

namespace nn::activation {

struct Sigmoid {
  [[nodiscard]] static nn::Scalar forward(nn::Scalar x) {
    return 1.F / (1.F + std::exp(-x));
  }

  [[nodiscard]] static nn::Scalar derivative_from_output(nn::Scalar y) { return y * (1.F - y); }
};

}  // namespace nn::activation
