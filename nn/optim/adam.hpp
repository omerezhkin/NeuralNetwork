#pragma once

#include "core/math/Linalg.hpp"
#include "nn/optim/any_optimizer_cache.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nn::optim {

struct AdamMatrixState {
  nn::MatrixXf m;
  nn::MatrixXf v;
};

struct AdamVectorState {
  nn::VectorXf m;
  nn::VectorXf v;
};

class Adam {
public:
  Adam(float learning_rate, float beta1 = 0.9F, float beta2 = 0.999F, float eps = 1e-8F)
      : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {
    if (learning_rate_ <= 0.F) {
      throw std::invalid_argument("Adam: learning_rate must be positive");
    }
    if (beta1_ < 0.F || beta1_ >= 1.F || beta2_ < 0.F || beta2_ >= 1.F) {
      throw std::invalid_argument("Adam: betas must be in [0, 1)");
    }
    if (eps_ <= 0.F) {
      throw std::invalid_argument("Adam: eps must be positive");
    }
  }

  void update_weights(nn::MatrixXf& param, const nn::MatrixXf& grad, AnyOptimizerCache& cache) {
    validate_same_shape(param, grad, "Adam::update_weights(Matrix)");
    AdamMatrixState* state = OptimizerCache_cast<AdamMatrixState>(&cache);
    if (!state) {
      cache = AnyOptimizerCache(AdamMatrixState{nn::MatrixXf::Zero(grad.rows(), grad.cols()),
                                                nn::MatrixXf::Zero(grad.rows(), grad.cols())});
      state = OptimizerCache_cast<AdamMatrixState>(&cache);
    }
    adam_step(param, grad, *state);
  }

  void update_weights(nn::VectorXf& param, const nn::VectorXf& grad, AnyOptimizerCache& cache) {
    validate_same_shape(param, grad, "Adam::update_weights(Vector)");
    AdamVectorState* state = OptimizerCache_cast<AdamVectorState>(&cache);
    if (!state) {
      cache = AnyOptimizerCache(AdamVectorState{nn::VectorXf::Zero(grad.size()),
                                                nn::VectorXf::Zero(grad.size())});
      state = OptimizerCache_cast<AdamVectorState>(&cache);
    }
    adam_step(param, grad, *state);
  }

  void iter_step() { ++t_; }

private:
  template<typename T, typename State>
  void adam_step(T& param, const T& grad, State& state) {
    const float step = static_cast<float>(t_ + 1);
    state.m = beta1_ * state.m + (1.F - beta1_) * grad;
    state.v = beta2_ * state.v + (1.F - beta2_) * grad.cwiseProduct(grad);

    const T m_hat = state.m / (1.F - std::pow(beta1_, step));
    const T v_hat = state.v / (1.F - std::pow(beta2_, step));

    param.array() -= learning_rate_ * m_hat.array() / (v_hat.array().sqrt() + eps_);
  }

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
  float beta1_;
  float beta2_;
  float eps_;
  std::uint64_t t_;
};

}  // namespace nn::optim
