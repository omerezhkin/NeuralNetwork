#pragma once

namespace nn::model {
class Network;
}

namespace nn::optim {

class Sgd {
public:
  explicit Sgd(float learning_rate);

  void apply_gradient_step(nn::model::Network& net);

private:
  float learning_rate_;
};

}  // namespace nn::optim
