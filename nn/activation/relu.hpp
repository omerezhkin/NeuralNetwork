#pragma once

#include "core/math/Linalg.hpp"

namespace nn::activation {

struct ReLU {
  [[nodiscard]] static nn::Scalar forward(nn::Scalar x) { return x > 0.F ? x : 0.F; }

  [[nodiscard]] static nn::Scalar derivative(nn::Scalar x) { return x > 0.F ? 1.F : 0.F; }
};

}  // namespace nn::activation
