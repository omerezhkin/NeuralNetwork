#pragma once

#include "core/math/Linalg.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"
#include "nn/layers/any_layer.hpp"

#include <cstddef>
#include <vector>

namespace nn::model {

class NetworkBuilder;

class Network {
public:
  nn::MatrixXf predict(nn::MatrixXf x);

  nn::MatrixXf forward(nn::MatrixXf x);

  nn::MatrixXf backward(nn::MatrixXf grad);

  void apply_gradients(float learning_rate);

  void zero_gradients();

  [[nodiscard]] std::size_t num_layers() const { return layers_.size(); }

private:
  friend class NetworkBuilder;

  explicit Network(std::vector<nn::layers::AnyLayer>&& layers);

  std::vector<nn::layers::AnyLayer> layers_;
  std::vector<nn::layers::AnyCache> caches_;
  std::vector<nn::layers::AnyGradients> grads_;
};

}  // namespace nn::model
