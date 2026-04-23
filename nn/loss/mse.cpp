#include "nn/loss/mse.hpp"

#include <stdexcept>

namespace nn::loss {

float Mse::value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const {
  if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
    throw std::invalid_argument("Mse::value: shape mismatch");
  }
  const float n = static_cast<float>(pred.size());
  return (pred - target).array().square().sum() / n;
}

nn::MatrixXf Mse::backward(const nn::MatrixXf& pred, const nn::MatrixXf& target) const {
  if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
    throw std::invalid_argument("Mse::backward: shape mismatch");
  }
  const float n = static_cast<float>(pred.size());
  return (2.F / n) * (pred - target);
}

}  // namespace nn::loss
