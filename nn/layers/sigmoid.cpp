#include "nn/layers/sigmoid.hpp"

#include <stdexcept>
#include <utility>

namespace nn::layers {

nn::MatrixXf Sigmoid::predict(const nn::MatrixXf& x) const {
  return (1.F / (1.F + (-x.array()).exp())).matrix();
}

std::pair<nn::MatrixXf, AnyCache> Sigmoid::forward(const nn::MatrixXf& x) const {
  nn::MatrixXf y = predict(x);
  return {y, AnyCache(Cache{y})};
}

std::pair<nn::MatrixXf, AnyGradients> Sigmoid::backward(const AnyCache& cache,
                                                         const nn::MatrixXf& dLdY) const {
  const Cache* typed_cache = AnyCache_cast<Cache>(&cache);
  if (!typed_cache) {
    throw std::bad_cast();
  }
  if (typed_cache->y.rows() != dLdY.rows() || typed_cache->y.cols() != dLdY.cols()) {
    throw std::invalid_argument("Sigmoid::backward: cache shape mismatch");
  }

  const auto y_array = typed_cache->y.array();
  nn::MatrixXf dLdX = (dLdY.array() * y_array * (1.F - y_array)).matrix();
  return {std::move(dLdX), AnyGradients(Gradients{})};
}

void Sigmoid::update(const AnyGradients& /*grads*/, float /*learning_rate*/) {
}

}  // namespace nn::layers