#pragma once

#include "core/math/Linalg.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"
#include "nn/layers/any_layer.hpp"
#include "nn/optim/any_optimizer.hpp"
#include "nn/optim/any_optimizer_cache.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nn::model {

class NetworkBuilder;

class Network {
public:
  [[nodiscard]] nn::MatrixXf predict(nn::MatrixXf x) const;

  [[nodiscard]] std::pair<nn::MatrixXf, std::vector<nn::layers::AnyCache>> forward(nn::MatrixXf x);

  [[nodiscard]] std::pair<nn::MatrixXf, std::vector<nn::layers::AnyGradients>> backward(
      nn::MatrixXf grad, const std::vector<nn::layers::AnyCache>& caches);

  void update(std::vector<nn::layers::AnyGradients>&& grads, nn::optim::AnyOptimizer& opt,
              std::vector<nn::optim::AnyOptimizerCache>& opt_caches);

  [[nodiscard]] std::size_t num_layers() const noexcept { return layers_.size(); }

private:
  friend class NetworkBuilder;

  explicit Network(std::vector<nn::layers::AnyLayer>&& layers);

  std::vector<nn::layers::AnyLayer> layers_;
};

inline Network::Network(std::vector<nn::layers::AnyLayer>&& layers) : layers_(std::move(layers)) {
  if (layers_.empty()) {
    throw std::invalid_argument("Network: layers must not be empty");
  }
}

inline nn::MatrixXf Network::predict(nn::MatrixXf x) const {
  for (const auto& layer : layers_) {
    x = layer.predict(x);
  }
  return x;
}

inline std::pair<nn::MatrixXf, std::vector<nn::layers::AnyCache>> Network::forward(nn::MatrixXf x) {
  std::vector<nn::layers::AnyCache> caches;
  caches.reserve(layers_.size());

  for (auto& layer : layers_) {
    auto [y, cache] = layer.forward(x);
    caches.push_back(std::move(cache));
    x = std::move(y);
  }
  return {std::move(x), std::move(caches)};
}

inline std::pair<nn::MatrixXf, std::vector<nn::layers::AnyGradients>> Network::backward(
    nn::MatrixXf grad, const std::vector<nn::layers::AnyCache>& caches) {
  if (caches.size() != layers_.size()) {
    throw std::invalid_argument("Network::backward: cache size mismatch");
  }

  std::vector<nn::layers::AnyGradients> grads;
  grads.resize(layers_.size());

  for (std::size_t i = layers_.size(); i-- > 0;) {
    auto [dLdX, layer_grads] = layers_[i].backward(caches[i], grad);
    grads[i] = std::move(layer_grads);
    grad = std::move(dLdX);
  }

  return {std::move(grad), std::move(grads)};
}

inline void Network::update(std::vector<nn::layers::AnyGradients>&& layer_grads,
                            nn::optim::AnyOptimizer& opt,
                            std::vector<nn::optim::AnyOptimizerCache>& opt_caches) {
  if (layer_grads.size() != layers_.size()) {
    throw std::invalid_argument("Network::update: gradients size mismatch");
  }
  if (opt_caches.size() != layers_.size()) {
    opt_caches.resize(layers_.size());
  }

  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layers_[i].update(layer_grads[i], opt, opt_caches[i]);
  }
}

}  // namespace nn::model
