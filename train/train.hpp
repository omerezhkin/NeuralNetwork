#pragma once

#include "core/math/Linalg.hpp"
#include "nn/loss/any_loss.hpp"
#include "nn/model/network.hpp"
#include "nn/optim/any_optimizer.hpp"
#include "nn/optim/any_optimizer_cache.hpp"
#include "train/data_loader.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nn::train {

class Trainer {
public:
  Trainer(nn::model::Network& model, nn::loss::AnyLoss& criterion, nn::optim::AnyOptimizer& optimizer);

  [[nodiscard]] float fit_epoch(std::uint64_t shuffle_seed = 0);

  [[nodiscard]] std::vector<float> fit(std::size_t epochs, std::uint64_t shuffle_seed = 0);

  void fit(const nn::MatrixXf& X, const nn::MatrixXf& Y, int epochs, int log_interval = 500);

  void set_data_loader(DataLoader* loader) { data_loader_ = loader; }

  void reset_optimizer_state() { opt_caches_.clear(); }

private:
  float train_batch(const nn::MatrixXf& batch_X, const nn::MatrixXf& batch_Y);

  nn::model::Network& model_;
  nn::loss::AnyLoss& criterion_;
  nn::optim::AnyOptimizer& optimizer_;
  DataLoader* data_loader_ = nullptr;
  std::vector<nn::optim::AnyOptimizerCache> opt_caches_;
};

inline Trainer::Trainer(nn::model::Network& model, nn::loss::AnyLoss& criterion,
                        nn::optim::AnyOptimizer& optimizer)
    : model_(model), criterion_(criterion), optimizer_(optimizer) {}

inline float Trainer::train_batch(const nn::MatrixXf& batch_X, const nn::MatrixXf& batch_Y) {
  auto [pred, caches] = model_.forward(batch_X);
  const float loss = criterion_.value(pred, batch_Y);
  nn::MatrixXf grad = criterion_.backward(pred, batch_Y);
  auto backward_result = model_.backward(std::move(grad), caches);
  model_.update(std::move(backward_result.second), optimizer_, opt_caches_);
  optimizer_.iter_step();
  return loss;
}

inline float Trainer::fit_epoch(std::uint64_t shuffle_seed) {
  if (!data_loader_) {
    throw std::logic_error("Trainer::fit_epoch: DataLoader not set");
  }
  if (shuffle_seed != 0) {
    data_loader_->reset_epoch(shuffle_seed);
  } else {
    data_loader_->reset_epoch();
  }

  float loss_sum = 0.F;
  std::size_t count = 0;
  for (auto it = data_loader_->begin(); it != data_loader_->end(); ++it) {
    auto [batch_X, batch_Y] = *it;
    loss_sum += train_batch(batch_X, batch_Y);
    ++count;
  }
  if (count == 0) {
    throw std::logic_error("Trainer::fit_epoch: no batches");
  }
  return loss_sum / static_cast<float>(count);
}

inline std::vector<float> Trainer::fit(std::size_t epochs, std::uint64_t shuffle_seed) {
  std::vector<float> history;
  history.reserve(epochs);
  for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
    const std::uint64_t seed = shuffle_seed != 0 ? shuffle_seed + epoch : 0;
    history.push_back(fit_epoch(seed));
  }
  return history;
}

inline void Trainer::fit(const nn::MatrixXf& X, const nn::MatrixXf& Y, int epochs,
                         int log_interval) {
  opt_caches_.clear();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    const float loss = train_batch(X, Y);

    if (epoch % log_interval == 0 || epoch == 1) {
      std::cout << "epoch " << epoch << "  loss = " << loss << '\n';
    }
  }
}

}  // namespace nn::train
