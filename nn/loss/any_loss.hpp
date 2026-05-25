#pragma once

#include "core/anymovable.hpp"
#include "core/math/Linalg.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nn::loss {

template<class Base>
class IAnyLoss : public Base {
public:
  virtual float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const = 0;

  virtual nn::MatrixXf backward(const nn::MatrixXf& pred, const nn::MatrixXf& target) const = 0;
};

template<class Base, class TObject>
class AnyLossModel : public Base {
  using CBase = Base;

public:
  using CBase::CBase;

  float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const override {
    return CBase::Object().value(pred, target);
  }

  nn::MatrixXf backward(const nn::MatrixXf& pred, const nn::MatrixXf& target) const override {
    return CBase::Object().backward(pred, target);
  }
};

class AnyLoss : public ::nn::library::CAnyMovable<IAnyLoss, AnyLossModel> {
  using CBase = ::nn::library::CAnyMovable<IAnyLoss, AnyLossModel>;

public:
  AnyLoss() = default;

  AnyLoss(AnyLoss&&) noexcept = default;
  AnyLoss& operator=(AnyLoss&&) noexcept = default;

  template<typename T, typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, AnyLoss>>>
  explicit AnyLoss(T&& loss) : CBase(std::forward<T>(loss)) {}

  template<typename T, typename... Args>
  explicit AnyLoss(std::in_place_type_t<T> tag, Args&&... args)
      : CBase(tag, std::forward<Args>(args)...) {}

  AnyLoss(const AnyLoss&) = delete;
  AnyLoss& operator=(const AnyLoss&) = delete;

  ~AnyLoss() = default;

  using CBase::clear;
  using CBase::emplace;
  using CBase::isDefined;
  using CBase::operator->;

  [[nodiscard]] explicit operator bool() const noexcept { return isDefined(); }

  [[nodiscard]] float value(const nn::MatrixXf& pred, const nn::MatrixXf& target) const {
    if (!isDefined()) {
      throw std::logic_error("AnyLoss: empty");
    }
    return (*this)->value(pred, target);
  }

  [[nodiscard]] nn::MatrixXf backward(const nn::MatrixXf& pred,
                                      const nn::MatrixXf& target) const {
    if (!isDefined()) {
      throw std::logic_error("AnyLoss: empty");
    }
    return (*this)->backward(pred, target);
  }
};

}  // namespace nn::loss
