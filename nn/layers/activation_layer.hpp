#pragma once

#include "core/math/Linalg.hpp"
#include "nn/activation/any_scalar_activation.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"
#include "nn/optim/any_optimizer.hpp"
#include "nn/optim/any_optimizer_cache.hpp"

#include <stdexcept>
#include <utility>

namespace nn::layers {

class ActivationLayer {
public:
  struct Cache {
    nn::MatrixXf x;
  };

  using Gradients = std::monostate;

  explicit ActivationLayer(nn::activation::AnyScalarActivation act) : act_(std::move(act)) {
    if (!act_.isDefined()) {
      throw std::invalid_argument("ActivationLayer: activation is empty");
    }
  }

  template<typename Activation>
  explicit ActivationLayer(Activation&& /*act*/)
      : act_(nn::activation::AnyScalarActivation(
            nn::activation::ScalarActivationFn<std::decay_t<Activation>>{})) {}

  [[nodiscard]] nn::MatrixXf predict(const nn::MatrixXf& x) const {
    validate_input(x, "predict");
    return x.unaryExpr([this](nn::Scalar v) { return act_.forward(v); });
  }

  [[nodiscard]] std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) const {
    validate_input(x, "forward");
    nn::MatrixXf y = predict(x);
    return {std::move(y), AnyCache(Cache{std::move(x)})};
  }

  [[nodiscard]] std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                                 const nn::MatrixXf& dLdY) const {
    const Cache* typed = AnyCache_cast<Cache>(&cache);
    if (!typed) {
      throw std::bad_cast();
    }
    if (typed->x.rows() != dLdY.rows() || typed->x.cols() != dLdY.cols()) {
      throw std::invalid_argument("ActivationLayer::backward: shape mismatch");
    }

    nn::MatrixXf dLdX = dLdY;
    for (Eigen::Index r = 0; r < typed->x.rows(); ++r) {
      for (Eigen::Index c = 0; c < typed->x.cols(); ++c) {
        const nn::Scalar x_val = typed->x(r, c);
        dLdX(r, c) = dLdY(r, c) * act_.derivative(x_val);
      }
    }
    return {std::move(dLdX), AnyGradients(Gradients{})};
  }

  void update(const AnyGradients&, nn::optim::AnyOptimizer&, nn::optim::AnyOptimizerCache&) {}

private:
  void validate_input(const nn::MatrixXf& x, const char* method) const {
    if (x.rows() == 0 || x.cols() == 0) {
      throw std::invalid_argument(std::string("ActivationLayer::") + method + ": empty input");
    }
  }

  nn::activation::AnyScalarActivation act_;
};

}  // namespace nn::layers
