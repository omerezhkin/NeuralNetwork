#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <memory>

namespace nn::random {

/// Потоковый ГПСЧ; реализация в random.cpp
class Rng {
public:
  Rng();
  explicit Rng(uint64_t seed);

  Rng(const Rng&) = delete;
  Rng& operator=(const Rng&) = delete;

  Rng(Rng&&) noexcept;
  Rng& operator=(Rng&&) noexcept;

  ~Rng();

  void seed(uint64_t s);

  [[nodiscard]] float uniform(float lo, float hi);

  [[nodiscard]] float normal(float mean = 0.F, float stddev = 1.F);

  void fill_uniform(Eigen::Ref<Eigen::MatrixXf> out, float lo, float hi);

  void fill_normal(Eigen::Ref<Eigen::MatrixXf> out, float mean, float stddev);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nn::random
