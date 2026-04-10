#pragma once

#include "core/anymovable.hpp"
#include "core/math/Linalg.hpp"
#include "nn/layers/any_cache.hpp"
#include "nn/layers/any_gradients.hpp"

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nn::layers {

/// Виртуальный интерфейс стёртого слоя. Входы/выходы и dL/dY — \c nn::MatrixXf (батч×признаки).
template<class Base>
class IAnyLayer : public Base {
public:
  virtual nn::MatrixXf predict(const nn::MatrixXf& x) = 0;

  virtual std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) = 0;

  virtual std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                         const nn::MatrixXf& dLdY) = 0;

  virtual void update(const AnyGradients& grads) = 0;
};

/// Модель: делегирует в TObject, упаковывает кэш/градиенты в Any*.
template<class Base, class TObject>
class AnyLayerModel : public Base {
  using CBase = Base;

public:
  using CBase::CBase;

  nn::MatrixXf predict(const nn::MatrixXf& x) override {
    return CBase::Object().predict(x);
  }

  std::pair<nn::MatrixXf, AnyCache> forward(const nn::MatrixXf& x) override {
    auto out = CBase::Object().forward(x);
    return {std::move(out.first), AnyCache(std::move(out.second))};
  }

  std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                 const nn::MatrixXf& dLdY) override {
    using ForwardRet =
        std::invoke_result_t<decltype(&TObject::forward), const TObject&, const nn::MatrixXf&>;
    using CacheType = std::tuple_element_t<1, ForwardRet>;
    const CacheType* c = AnyCache_cast<CacheType>(&cache);
    if (!c) {
      throw std::bad_cast();
    }
    auto out = CBase::Object().backward(*c, dLdY);
    return {std::move(out.first), AnyGradients(std::move(out.second))};
  }

  void update(const AnyGradients& grads) override {
    using ForwardRet =
        std::invoke_result_t<decltype(&TObject::forward), const TObject&, const nn::MatrixXf&>;
    using CacheType = std::tuple_element_t<1, ForwardRet>;
    using BackwardRet =
        std::invoke_result_t<decltype(&TObject::backward), const TObject&, const CacheType&,
                             const nn::MatrixXf&>;
    using GradType = std::tuple_element_t<1, BackwardRet>;
    const GradType* g = AnyGradients_cast<GradType>(&grads);
    if (!g) {
      throw std::bad_cast();
    }
    CBase::Object().update(*g);
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

  nn::MatrixXf predict(const nn::MatrixXf& x) {
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

  std::pair<nn::MatrixXf, AnyGradients> backward(const AnyCache& cache,
                                                 const nn::MatrixXf& dLdY) {
    if (!isDefined()) {
      throw std::logic_error("AnyLayer: empty");
    }
    return (*this)->backward(cache, dLdY);
  }

  void update(const AnyGradients& grads) {
    if (!isDefined()) {
      throw std::logic_error("AnyLayer: empty");
    }
    (*this)->update(grads);
  }
};

}  // namespace nn::layers
