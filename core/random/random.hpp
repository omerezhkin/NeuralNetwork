#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace nn::random {

enum class InitScheme { Normal, Xavier, He };

struct Gain {
  float value = 1.F;
};

/// Потоковый ГПСЧ (header-only).
class Rng {
public:
  Rng() = default;

  explicit Rng(uint64_t s) { reseed(s); }

  void reseed(uint64_t s) { engine_.seed(static_cast<std::mt19937::result_type>(s)); }

  [[nodiscard]] float uniform(float lo, float hi) {
    const float u = uniform_dist_(engine_);
    return lo + u * (hi - lo);
  }

  [[nodiscard]] float normal(float mean = 0.F, float stddev = 1.F) {
    return mean + stddev * normal_dist_(engine_);
  }

  void fill_uniform(Eigen::Ref<Eigen::MatrixXf> out, float lo, float hi) {
    for (Eigen::Index i = 0; i < out.size(); ++i) {
      out.data()[i] = uniform(lo, hi);
    }
  }

  void fill_normal(Eigen::Ref<Eigen::MatrixXf> out, float mean, float stddev) {
    for (Eigen::Index i = 0; i < out.size(); ++i) {
      out.data()[i] = normal(mean, stddev);
    }
  }

  void init_linear_weights(Eigen::Ref<Eigen::MatrixXf> W, Eigen::Index in_features,
                           Eigen::Index out_features, InitScheme scheme, Gain gain = Gain{}) {
    if (in_features <= 0 || out_features <= 0) {
      throw std::invalid_argument("Rng::init_linear_weights: dimensions must be positive");
    }

    float stddev = 1.F;
    switch (scheme) {
      case InitScheme::Normal:
        stddev = gain.value;
        break;
      case InitScheme::Xavier:
        stddev = std::sqrt(2.F / static_cast<float>(in_features + out_features));
        break;
      case InitScheme::He:
        stddev = std::sqrt(2.F / static_cast<float>(in_features));
        break;
    }
    fill_normal(W, 0.F, stddev);
  }

private:
  std::mt19937 engine_{std::random_device{}()};
  std::uniform_real_distribution<float> uniform_dist_{0.F, 1.F};
  std::normal_distribution<float> normal_dist_{0.F, 1.F};
};

}  // namespace nn::random
