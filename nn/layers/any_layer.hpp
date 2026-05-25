#pragma once

#include "core/anymovable.hpp"
#include "core/math/Linalg.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"
#include "nn/optim/any_optimizer.hpp"
#include "nn/optim/any_optimizer_cache.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nn::layers {

template<class Base>
class IAnyLayer : public Base {
public:
  virtual nn::MatrixXf predict(const nn::MatrixXf& x) const = 0;

  virtual std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) = 0;

  virtual std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                         const nn::MatrixXf& dLdY) = 0;

  virtual void update(const AnyGradients& grads, nn::optim::AnyOptimizer& opt,
                      nn::optim::AnyOptimizerCache& cache) = 0;
};

template<class Base, class TObject>
class AnyLayerModel : public Base {
  using CBase = Base;

public:
  using CBase::CBase;

  nn::MatrixXf predict(const nn::MatrixXf& x) const override {
    return CBase::Object().predict(x);
  }

  std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) override {
    return CBase::Object().forward(x);
  }

  std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                 const nn::MatrixXf& dLdY) override {
    return CBase::Object().backward(cache, dLdY);
  }

  void update(const AnyGradients& grads, nn::optim::AnyOptimizer& opt,
              nn::optim::AnyOptimizerCache& cache) override {
    CBase::Object().update(grads, opt, cache);
  }
};

class AnyLayer : public ::nn::library::CAnyMovable<IAnyLayer, AnyLayerModel> {
  using CBase = ::nn::library::CAnyMovable<IAnyLayer, AnyLayerModel>;

public:
  AnyLayer() = default;

  AnyLayer(AnyLayer&&) noexcept = default;
  AnyLayer& operator=(AnyLayer&&) noexcept = default;

  template<typename T,
           typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, AnyLayer>>>
  explicit AnyLayer(T&& layer) : CBase(std::forward<T>(layer)) {}

  template<typename T, typename... Args>
  explicit AnyLayer(std::in_place_type_t<T> tag, Args&&... args)
      : CBase(tag, std::forward<Args>(args)...) {}

  AnyLayer(const AnyLayer&) = delete;
  AnyLayer& operator=(const AnyLayer&) = delete;

  ~AnyLayer() = default;

  using CBase::clear;
  using CBase::emplace;
  using CBase::isDefined;
  using CBase::operator->;

  [[nodiscard]] explicit operator bool() const noexcept { return isDefined(); }

  nn::MatrixXf predict(const nn::MatrixXf& x) const {
    if (!isDefined()) {
      throw std::logic_error("AnyLayer: empty");
    }
    return (*this)->predict(x);
  }

  std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) {
    if (!isDefined()) {
      throw std::logic_error("AnyLayer: empty");
    }
    return (*this)->forward(x);
  }

  std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache, const nn::MatrixXf& dLdY) {
    if (!isDefined()) {
      throw std::logic_error("AnyLayer: empty");
    }
    return (*this)->backward(cache, dLdY);
  }

  void update(const AnyGradients& grads, nn::optim::AnyOptimizer& opt,
              nn::optim::AnyOptimizerCache& cache) {
    if (!isDefined()) {
      throw std::logic_error("AnyLayer: empty");
    }
    (*this)->update(grads, opt, cache);
  }
};

}  // namespace nn::layers
