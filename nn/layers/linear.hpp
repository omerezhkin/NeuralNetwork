#pragma once

#include "core/math/Linalg.hpp"
#include "core/random/random.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"

#include <utility>

namespace nn::layers {

class Linear {
public:
  /// Исходный батч X.
  struct Cache {
    nn::MatrixXf x;
  };

  /// Градиенты параметров линейного слоя.
  struct Gradients {
    nn::MatrixXf dW;
    nn::VectorXf db;
  };

  Linear(nn::In in_features, nn::Out out_features, nn::random::Rng& rnd);

  [[nodiscard]] nn::MatrixXf predict(const nn::MatrixXf& x) const;

  [[nodiscard]] std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) const;

  [[nodiscard]] std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                               const nn::MatrixXf& dLdY) const;

  void update(const AnyGradients& grads, float learning_rate);

private:
  Eigen::Index in_features_;
  Eigen::Index out_features_;
  nn::MatrixXf W_;
  nn::VectorXf b_;
};

}  // namespace nn::layers
