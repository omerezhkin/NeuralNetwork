#pragma once

#include "core/math/Linalg.hpp"

#include <cmath>

namespace nn::activation {

struct ReLU {
  [[nodiscard]] static nn::Scalar forward(nn::Scalar x) { return x > 0.F ? x : 0.F; }

  [[nodiscard]] static nn::Scalar derivative(nn::Scalar x) { return x > 0.F ? 1.F : 0.F; }
};

struct Sigmoid {
  [[nodiscard]] static nn::Scalar forward(nn::Scalar x) {
    return 1.F / (1.F + std::exp(-x));
  }

  [[nodiscard]] static nn::Scalar derivative_from_output(nn::Scalar y) { return y * (1.F - y); }
};

struct Tanh {
  [[nodiscard]] static nn::Scalar forward(nn::Scalar x) { return std::tanh(x); }

  [[nodiscard]] static nn::Scalar derivative_from_output(nn::Scalar y) {
    return 1.F - y * y;
  }
};

}  // namespace nn::activation
