#pragma once

#include "core/math/Linalg.hpp"

#include <stdexcept>

namespace nn::loss {

inline void validate_same_shape(const nn::MatrixXf& pred, const nn::MatrixXf& target,
                                const char* ctx) {
  if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
    throw std::invalid_argument(std::string(ctx) + ": shape mismatch");
  }
  if (pred.size() == 0) {
    throw std::invalid_argument(std::string(ctx) + ": empty tensors");
  }
}

struct Mse {
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
};

struct Mae {
  [[nodiscard]] float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Mae::value");
    const float n = static_cast<float>(pred.size());
    return (pred - target).cwiseAbs().sum() / n;
  }

  [[nodiscard]] nn::MatrixXf backward(const nn::MatrixXf& pred,
                                      const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Mae::backward");
    const float n = static_cast<float>(pred.size());
    return (pred - target).cwiseSign() / n;
  }
};

struct Huber {
  float delta = 1.F;

  [[nodiscard]] float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Huber::value");
    if (delta <= 0.F) {
      throw std::invalid_argument("Huber::value: delta must be positive");
    }
    const float n = static_cast<float>(pred.size());
    const nn::MatrixXf diff = pred - target;
    const auto abs_diff = diff.cwiseAbs().array();
    const auto loss = (abs_diff <= delta)
                          .select(0.5F * diff.array().square(), delta * (abs_diff - 0.5F * delta));
    return loss.sum() / n;
  }

  [[nodiscard]] nn::MatrixXf backward(const nn::MatrixXf& pred,
                                       const nn::MatrixXf& target) const {
    validate_same_shape(pred, target, "Huber::backward");
    if (delta <= 0.F) {
      throw std::invalid_argument("Huber::backward: delta must be positive");
    }
    const float n = static_cast<float>(pred.size());
    const nn::MatrixXf diff = pred - target;
    const auto abs_diff = diff.cwiseAbs().array();
    return (abs_diff <= delta)
               .select(diff.array(), delta * diff.cwiseSign().array())
               .matrix() /
           n;
  }
};

}  // namespace nn::loss
