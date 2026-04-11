#include "nn/layers/linear.hpp"

#include <stdexcept>
#include <utility>

namespace nn::layers {

Linear::Linear(nn::In in_features, nn::Out out_features, nn::random::Rng& rnd)
    : in_features_(in_features.value),
      out_features_(out_features.value),
      W_(in_features_, out_features_),
      b_(out_features_) {
  if (in_features_ <= 0 || out_features_ <= 0) {
    throw std::invalid_argument("Linear: in/out features must be positive");
  }

  rnd.fill_normal(W_, 0.F, 1.F);

  nn::MatrixXf b_as_matrix(1, out_features_);
  rnd.fill_normal(b_as_matrix, 0.F, 1.F);
  b_ = b_as_matrix.transpose();
}

nn::MatrixXf Linear::predict(const nn::MatrixXf& x) const {
  if (x.cols() != in_features_) {
    throw std::invalid_argument("Linear::predict: bad input feature size");
  }

  nn::MatrixXf y = x * W_;
  y.rowwise() += b_.transpose();
  return y;
}

std::pair<nn::MatrixXf, AnyCache> Linear::forward(const nn::MatrixXf& x) const {
  nn::MatrixXf y = predict(x);
  return {std::move(y), AnyCache(Cache{x})};
}

std::pair<nn::MatrixXf, AnyGradients> Linear::backward(const AnyCache& cache,
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

void Linear::update(const AnyGradients& grads, float learning_rate) {
  const Gradients* typed_grads = AnyGradients_cast<Gradients>(&grads);
  if (!typed_grads) {
    throw std::bad_cast();
  }
  W_ -= learning_rate * typed_grads->dW;
  b_ -= learning_rate * typed_grads->db;
}

}  // namespace nn::layers
