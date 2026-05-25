#pragma once

#include "core/anymovable.hpp"
#include "core/math/Linalg.hpp"
#include "nn/optim/any_optimizer_cache.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nn::optim {

template<class Base>
class IAnyOptimizer : public Base {
public:
  virtual void update_weights(nn::MatrixXf& param, const nn::MatrixXf& grad,
                              AnyOptimizerCache& cache) = 0;

  virtual void update_weights(nn::VectorXf& param, const nn::VectorXf& grad,
                              AnyOptimizerCache& cache) = 0;

  virtual void iter_step() {}
};

template<class Base, class TObject>
class AnyOptimizerModel : public Base {
  using CBase = Base;

public:
  using CBase::CBase;

  void update_weights(nn::MatrixXf& param, const nn::MatrixXf& grad,
                      AnyOptimizerCache& cache) override {
    CBase::Object().update_weights(param, grad, cache);
  }

  void update_weights(nn::VectorXf& param, const nn::VectorXf& grad,
                      AnyOptimizerCache& cache) override {
    CBase::Object().update_weights(param, grad, cache);
  }

  void iter_step() override { CBase::Object().iter_step(); }
};

class AnyOptimizer : public ::nn::library::CAnyMovable<IAnyOptimizer, AnyOptimizerModel> {
  using CBase = ::nn::library::CAnyMovable<IAnyOptimizer, AnyOptimizerModel>;

public:
  AnyOptimizer() = default;

  AnyOptimizer(AnyOptimizer&&) noexcept = default;
  AnyOptimizer& operator=(AnyOptimizer&&) noexcept = default;

  template<typename T,
           typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, AnyOptimizer>>>
  explicit AnyOptimizer(T&& opt) : CBase(std::forward<T>(opt)) {}

  template<typename T, typename... Args>
  explicit AnyOptimizer(std::in_place_type_t<T> tag, Args&&... args)
      : CBase(tag, std::forward<Args>(args)...) {}

  AnyOptimizer(const AnyOptimizer&) = delete;
  AnyOptimizer& operator=(const AnyOptimizer&) = delete;

  ~AnyOptimizer() = default;

  using CBase::clear;
  using CBase::emplace;
  using CBase::isDefined;
  using CBase::operator->;

  [[nodiscard]] explicit operator bool() const noexcept { return isDefined(); }

  void update_weights(nn::MatrixXf& param, const nn::MatrixXf& grad, AnyOptimizerCache& cache) {
    if (!isDefined()) {
      throw std::logic_error("AnyOptimizer: empty");
    }
    (*this)->update_weights(param, grad, cache);
  }

  void update_weights(nn::VectorXf& param, const nn::VectorXf& grad, AnyOptimizerCache& cache) {
    if (!isDefined()) {
      throw std::logic_error("AnyOptimizer: empty");
    }
    (*this)->update_weights(param, grad, cache);
  }

  void iter_step() {
    if (isDefined()) {
      (*this)->iter_step();
    }
  }
};

}  // namespace nn::optim
