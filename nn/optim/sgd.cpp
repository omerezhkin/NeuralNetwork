#include "nn/optim/sgd.hpp"

#include "nn/model/network.hpp"

namespace nn::optim {

Sgd::Sgd(float learning_rate) : learning_rate_(learning_rate) {}

void Sgd::apply_gradient_step(nn::model::Network& net) {
  net.apply_gradients(learning_rate_);
  net.zero_gradients();
}

}  // namespace nn::optim
