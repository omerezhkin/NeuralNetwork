#pragma once

#include "core/math/Linalg.hpp"

#include <stdexcept>
#include <string>

namespace nn::loss {

class Mse {
public:
  [[nodiscard]] float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Mse::value");
    const float n = static_cast<float>(pred.size());
    return (pred - target).array().square().sum() / n;
  }

  [[nodiscard]] nn::MatrixXf backward(const nn::MatrixXf& pred,
                                      const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Mse::backward");
    const float n = static_cast<float>(pred.size());
    return (2.F / n) * (pred - target);
  }

private:
  static void validate_same_shape(const nn::MatrixXf& pred, const nn::MatrixXf& target,
                                  const char* ctx) {
    if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
      throw std::invalid_argument(std::string(ctx) + ": shape mismatch");
    }
    if (pred.size() == 0) {
      throw std::invalid_argument(std::string(ctx) + ": empty tensors");
    }
  }
};

}  // namespace nn::loss
