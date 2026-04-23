#pragma once

#include "core/math/Linalg.hpp"

namespace nn::loss {

class Mse {
public:
  [[nodiscard]] float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const;

  [[nodiscard]] nn::MatrixXf backward(const nn::MatrixXf& pred, const nn::MatrixXf& target) const;
};

}  // namespace nn::loss
