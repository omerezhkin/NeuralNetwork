#pragma once

#include "core/math/Linalg.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"

#include <utility>
#include <variant>

namespace nn::layers {

class Sigmoid {
public:
  /// Выход активации y.
  struct Cache {
    nn::MatrixXf y;
  };

  using Gradients = std::monostate;

  [[nodiscard]] nn::MatrixXf predict(const nn::MatrixXf& x) const;

  [[nodiscard]] std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) const;

  [[nodiscard]] std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                               const nn::MatrixXf& dLdY) const;

  void update(const AnyGradients& grads, float learning_rate);
};

}  // namespace nn::layers
