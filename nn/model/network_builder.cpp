#include "nn/model/network_builder.hpp"

#include "nn/layers/linear.hpp"
#include "nn/layers/sigmoid.hpp"

#include <utility>

namespace nn::model {

NetworkBuilder::NetworkBuilder(uint32_t seed) : rng_(seed) {}

NetworkBuilder& NetworkBuilder::add_linear(nn::In in_features, nn::Out out_features) {
  layers_.emplace_back(nn::layers::Linear(in_features, out_features, rng_));
  return *this;
}

NetworkBuilder& NetworkBuilder::add_sigmoid() {
  layers_.emplace_back(nn::layers::Sigmoid{});
  return *this;
}

Network NetworkBuilder::build() {
  return Network(std::move(layers_));
}

}  // namespace nn::model
