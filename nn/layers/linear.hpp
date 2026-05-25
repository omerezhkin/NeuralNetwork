#pragma once

#include "core/math/Linalg.hpp"
#include "core/random/random.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"
#include "nn/optim/any_optimizer.hpp"
#include "nn/optim/any_optimizer_cache.hpp"

#include <stdexcept>
#include <utility>

namespace nn::layers {

class Linear {
public:
  struct Cache {
    nn::MatrixXf x;
  };

  struct Gradients {
    nn::MatrixXf dW;
    nn::VectorXf db;
  };

  struct LinearOptCache {
    nn::optim::AnyOptimizerCache w;
    nn::optim::AnyOptimizerCache b;
  };

  Linear(nn::In in_features, nn::Out out_features, nn::random::Rng& rnd,
         nn::random::InitScheme scheme = nn::random::InitScheme::Xavier,
         nn::random::Gain gain = nn::random::Gain{});

  [[nodiscard]] nn::MatrixXf predict(const nn::MatrixXf& x) const;

  [[nodiscard]] std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) const;

  [[nodiscard]] std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                               const nn::MatrixXf& dLdY) const;

  void update(const AnyGradients& grads, nn::optim::AnyOptimizer& opt,
              nn::optim::AnyOptimizerCache& cache);

private:
  Eigen::Index in_features_;
  Eigen::Index out_features_;
  nn::MatrixXf W_;
  nn::VectorXf b_;
};

inline Linear::Linear(nn::In in_features, nn::Out out_features, nn::random::Rng& rnd,
                      nn::random::InitScheme scheme, nn::random::Gain gain)
    : in_features_(in_features.value),
      out_features_(out_features.value),
      W_(in_features_, out_features_),
      b_(nn::VectorXf::Zero(out_features_)) {
  if (in_features_ <= 0 || out_features_ <= 0) {
    throw std::invalid_argument("Linear: in/out features must be positive");
  }

  rnd.init_linear_weights(W_, in_features_, out_features_, scheme, gain);
}

inline nn::MatrixXf Linear::predict(const nn::MatrixXf& x) const {
  if (x.cols() != in_features_) {
    throw std::invalid_argument("Linear::predict: bad input feature size");
  }

  nn::MatrixXf y = x * W_;
  y.rowwise() += b_.transpose();
  return y;
}

inline std::pair<nn::MatrixXf, AnyCache> Linear::forward(const nn::MatrixXf& x) const {
  nn::MatrixXf y = predict(x);
  return {std::move(y), AnyCache(Cache{x})};
}

inline std::pair<nn::MatrixXf, AnyGradients> Linear::backward(const AnyCache& cache,
                                                              const nn::MatrixXf& dLdY) const {
  const Cache* typed_cache = AnyCache_cast<Cache>(&cache);
  if (!typed_cache) {
    throw std::bad_cast();
  }
  if (typed_cache->x.rows() != dLdY.rows()) {
    throw std::invalid_argument("Linear::backward: cache batch size mismatch");
  }
  if (typed_cache->x.cols() != in_features_) {
    throw std::invalid_argument("Linear::backward: cache feature size mismatch");
  }
  if (dLdY.cols() != out_features_) {
    throw std::invalid_argument("Linear::backward: bad dLdY feature size");
  }

  nn::MatrixXf dLdX = dLdY * W_.transpose();
  nn::MatrixXf dW = typed_cache->x.transpose() * dLdY;
  nn::VectorXf db = dLdY.colwise().sum().transpose();

  return {std::move(dLdX), AnyGradients(Gradients{std::move(dW), std::move(db)})};
}

inline void Linear::update(const AnyGradients& grads, nn::optim::AnyOptimizer& opt,
                           nn::optim::AnyOptimizerCache& cache) {
  const Gradients* typed_grads = nn::layers::AnyGradients_cast<Gradients>(&grads);
  if (!typed_grads) {
    throw std::bad_cast();
  }

  LinearOptCache* typed_cache = nn::optim::OptimizerCache_cast<LinearOptCache>(&cache);
  if (!typed_cache) {
    cache = nn::optim::AnyOptimizerCache(LinearOptCache{});
    typed_cache = nn::optim::OptimizerCache_cast<LinearOptCache>(&cache);
  }

  opt.update_weights(W_, typed_grads->dW, typed_cache->w);
  opt.update_weights(b_, typed_grads->db, typed_cache->b);
}

}  // namespace nn::layers
