#pragma once

#include "core/math/Linalg.hpp"
#include "core/random/random.hpp"
#include "nn/activation/relu.hpp"
#include "nn/activation/sigmoid.hpp"
#include "nn/activation/tanh.hpp"
#include "nn/layers/activation_layer.hpp"
#include "nn/layers/any_layer.hpp"
#include "nn/layers/linear.hpp"
#include "nn/model/network.hpp"

#include <cstdint>
#include <vector>

namespace nn::model {

class NetworkBuilder {
public:
  explicit NetworkBuilder(uint32_t seed = 42);

  NetworkBuilder& add_linear(nn::In in_features, nn::Out out_features,
                             nn::random::InitScheme scheme = nn::random::InitScheme::Xavier);

  NetworkBuilder& add_relu();

  NetworkBuilder& add_sigmoid();

  NetworkBuilder& add_tanh();

  template<typename Activation>
  NetworkBuilder& add_activation(Activation&& act) {
    layers_.emplace_back(nn::layers::ActivationLayer(std::forward<Activation>(act)));
    return *this;
  }

  [[nodiscard]] Network build();

private:
  nn::random::Rng rng_;
  std::vector<nn::layers::AnyLayer> layers_;
};

inline NetworkBuilder::NetworkBuilder(uint32_t seed) : rng_(seed) {}

inline NetworkBuilder& NetworkBuilder::add_linear(nn::In in_features, nn::Out out_features,
                                                  nn::random::InitScheme scheme) {
  layers_.emplace_back(nn::layers::Linear(in_features, out_features, rng_, scheme));
  return *this;
}

inline NetworkBuilder& NetworkBuilder::add_relu() {
  return add_activation(nn::activation::ReLU{});
}

inline NetworkBuilder& NetworkBuilder::add_sigmoid() {
  return add_activation(nn::activation::Sigmoid{});
}

inline NetworkBuilder& NetworkBuilder::add_tanh() {
  return add_activation(nn::activation::Tanh{});
}

inline Network NetworkBuilder::build() { return Network(std::move(layers_)); }

}  // namespace nn::model
