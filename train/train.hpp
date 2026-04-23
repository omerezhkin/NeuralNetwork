#pragma once

#include "core/math/Linalg.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network.hpp"
#include "nn/optim/sgd.hpp"

namespace nn::train {

class Trainer {
public:
  Trainer(nn::model::Network& model, nn::loss::Mse& criterion, nn::optim::Sgd& optimizer);

  void fit(const nn::MatrixXf& X, const nn::MatrixXf& Y, int epochs, int log_interval = 500);

private:
  nn::model::Network& model_;
  nn::loss::Mse& criterion_;
  nn::optim::Sgd& optimizer_;
};

}  // namespace nn::train
