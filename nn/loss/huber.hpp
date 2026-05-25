#pragma once

#include "core/math/Linalg.hpp"

#include <stdexcept>
#include <string>

namespace nn::loss {

class Huber {
public:
  explicit Huber(float delta = 1.F) : delta_(delta) {
    if (delta_ <= 0.F) {
      throw std::invalid_argument("Huber: delta must be positive");
    }
  }

  [[nodiscard]] float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Huber::value");
    const float n = static_cast<float>(pred.size());
    const nn::MatrixXf diff = pred - target;
    const auto abs_diff = diff.cwiseAbs().array();
    const auto loss = (abs_diff <= delta_)
                          .select(0.5F * diff.array().square(),
                                  delta_ * (abs_diff - 0.5F * delta_));
    return loss.sum() / n;
  }

  [[nodiscard]] nn::MatrixXf backward(const nn::MatrixXf& pred,
                                      const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Huber::backward");
    const float n = static_cast<float>(pred.size());
    const nn::MatrixXf diff = pred - target;
    const auto abs_diff = diff.cwiseAbs().array();
    return (abs_diff <= delta_)
               .select(diff.array(), delta_ * diff.cwiseSign().array())
               .matrix() /
           n;
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

  float delta_;
};

}  // namespace nn::loss
