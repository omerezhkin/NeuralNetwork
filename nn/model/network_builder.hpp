#pragma once

#include "core/math/Linalg.hpp"
#include "core/random/random.hpp"
#include "nn/layers/any_layer.hpp"
#include "nn/model/network.hpp"

#include <cstdint>
#include <vector>

namespace nn::model {

class NetworkBuilder {
public:
  explicit NetworkBuilder(uint32_t seed = 42);

  NetworkBuilder& add_linear(nn::In in_features, nn::Out out_features);

  NetworkBuilder& add_sigmoid();

  Network build();

private:
  nn::random::Rng rng_;
  std::vector<nn::layers::AnyLayer> layers_;
};

}  // namespace nn::model
