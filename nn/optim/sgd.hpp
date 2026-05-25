#pragma once

#include "core/math/Linalg.hpp"
#include "nn/optim/any_optimizer_cache.hpp"

#include <stdexcept>
#include <string>

namespace nn::optim {

class Sgd {
public:
  explicit Sgd(float learning_rate) : learning_rate_(learning_rate) {
    if (learning_rate_ <= 0.F) {
      throw std::invalid_argument("Sgd: learning_rate must be positive");
    }
  }

  void update_weights(nn::MatrixXf& param, const nn::MatrixXf& grad, AnyOptimizerCache&) {
    validate_same_shape(param, grad, "Sgd::update_weights(Matrix)");
    param -= learning_rate_ * grad;
  }

  void update_weights(nn::VectorXf& param, const nn::VectorXf& grad, AnyOptimizerCache&) {
    validate_same_shape(param, grad, "Sgd::update_weights(Vector)");
    param -= learning_rate_ * grad;
  }

  void iter_step() {}

private:
  static void validate_same_shape(const nn::MatrixXf& param, const nn::MatrixXf& grad,
                                  const char* ctx) {
    if (param.rows() != grad.rows() || param.cols() != grad.cols()) {
      throw std::invalid_argument(std::string(ctx) + ": shape mismatch");
    }
  }

  static void validate_same_shape(const nn::VectorXf& param, const nn::VectorXf& grad,
                                  const char* ctx) {
    if (param.size() != grad.size()) {
      throw std::invalid_argument(std::string(ctx) + ": shape mismatch");
    }
  }

  float learning_rate_;
};

}  // namespace nn::optim
