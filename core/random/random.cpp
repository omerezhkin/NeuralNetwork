#include "core/random/random.hpp"

#include <random>

namespace nn::random {

struct Rng::Impl {
  std::mt19937 engine{std::random_device{}()};
  std::uniform_real_distribution<float> uniform_dist{0.F, 1.F};
  std::normal_distribution<float> normal_dist{0.F, 1.F};
};

Rng::Rng() : impl_(std::make_unique<Impl>()) {}

Rng::Rng(uint64_t seed) : impl_(std::make_unique<Impl>()) { this->seed(seed); }

Rng::Rng(Rng&&) noexcept = default;
Rng& Rng::operator=(Rng&&) noexcept = default;

Rng::~Rng() = default;

void Rng::seed(uint64_t s) { impl_->engine.seed(static_cast<std::mt19937::result_type>(s)); }

float Rng::uniform(float lo, float hi) {
  const float u = impl_->uniform_dist(impl_->engine);
  return lo + u * (hi - lo);
}

float Rng::normal(float mean, float stddev) {
  return mean + stddev * impl_->normal_dist(impl_->engine);
}

void Rng::fill_uniform(Eigen::Ref<Eigen::MatrixXf> out, float lo, float hi) {
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out.data()[i] = uniform(lo, hi);
  }
}

void Rng::fill_normal(Eigen::Ref<Eigen::MatrixXf> out, float mean, float stddev) {
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out.data()[i] = normal(mean, stddev);
  }
}

}  // namespace nn::random
