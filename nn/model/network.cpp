#include "nn/model/network.hpp"

#include <stdexcept>
#include <utility>

namespace nn::model {

Network::Network(std::vector<nn::layers::AnyLayer>&& layers)
    : layers_(std::move(layers)) {
  if (layers_.empty()) {
    throw std::invalid_argument("Network: layers must not be empty");
  }
}

nn::MatrixXf Network::predict(nn::MatrixXf x) {
  for (auto& layer : layers_) {
    x = layer.predict(x);
  }
  return x;
}

nn::MatrixXf Network::forward(nn::MatrixXf x) {
  const auto n = layers_.size();
  caches_.clear();
  caches_.reserve(n);

  for (auto& layer : layers_) {
    auto [y, cache] = layer.forward(x);
    caches_.push_back(std::move(cache));
    x = std::move(y);
  }
  return x;
}

nn::MatrixXf Network::backward(nn::MatrixXf grad) {
  if (caches_.size() != layers_.size()) {
    throw std::logic_error("Network::backward: call forward first");
  }

  const auto n = layers_.size();
  grads_.clear();
  grads_.resize(n);

  for (std::size_t i = n; i-- > 0;) {
    auto [dLdX, layer_grads] = layers_[i].backward(caches_[i], grad);
    grads_[i] = std::move(layer_grads);
    grad = std::move(dLdX);
  }

  caches_.clear();
  return grad;
}

void Network::apply_gradients(float learning_rate) {
  if (grads_.size() != layers_.size()) {
    throw std::logic_error("Network::apply_gradients: call backward first");
  }
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    layers_[i].update(grads_[i], learning_rate);
  }
}

void Network::zero_gradients() {
  grads_.clear();
}

}  // namespace nn::model
