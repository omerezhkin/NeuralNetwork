#include "train/train.hpp"

#include <iostream>
#include <utility>

namespace nn::train {

Trainer::Trainer(nn::model::Network& model, nn::loss::Mse& criterion, nn::optim::Sgd& optimizer)
    : model_(model), criterion_(criterion), optimizer_(optimizer) {}

void Trainer::fit(const nn::MatrixXf& X, const nn::MatrixXf& Y, int epochs, int log_interval) {
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    nn::MatrixXf pred = model_.forward(X);

    if (epoch % log_interval == 0 || epoch == 1) {
      std::cout << "epoch " << epoch
                << "  loss = " << criterion_.value(pred, Y) << '\n';
    }

    nn::MatrixXf grad = criterion_.backward(pred, Y);
    model_.backward(std::move(grad));
    optimizer_.apply_gradient_step(model_);
  }
}

}  // namespace nn::train
