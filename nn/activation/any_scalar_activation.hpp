#pragma once

#include "core/anymovable.hpp"
#include "core/math/Linalg.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nn::activation {

template<class Base>
class IAnyScalarActivation : public Base {
public:
  virtual nn::Scalar forward(nn::Scalar x) const = 0;
  virtual nn::Scalar derivative(nn::Scalar x) const = 0;
  virtual nn::Scalar derivative_from_output(nn::Scalar y) const = 0;
};

template<class Base, class TObject>
class AnyScalarActivationModel : public Base {
  using CBase = Base;

public:
  using CBase::CBase;

  nn::Scalar forward(nn::Scalar x) const override { return CBase::Object().forward(x); }

  nn::Scalar derivative(nn::Scalar x) const override { return CBase::Object().derivative(x); }

  nn::Scalar derivative_from_output(nn::Scalar y) const override {
    return CBase::Object().derivative_from_output(y);
  }
};

class AnyScalarActivation
    : public ::nn::library::CAnyMovable<IAnyScalarActivation, AnyScalarActivationModel> {
  using CBase = ::nn::library::CAnyMovable<IAnyScalarActivation, AnyScalarActivationModel>;

public:
  AnyScalarActivation() = default;

  AnyScalarActivation(AnyScalarActivation&&) noexcept = default;
  AnyScalarActivation& operator=(AnyScalarActivation&&) noexcept = default;

  template<typename T, typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>,
                                                                   AnyScalarActivation>>>
  explicit AnyScalarActivation(T&& act) : CBase(std::forward<T>(act)) {}

  template<typename T, typename... Args>
  explicit AnyScalarActivation(std::in_place_type_t<T> tag, Args&&... args)
      : CBase(tag, std::forward<Args>(args)...) {}

  AnyScalarActivation(const AnyScalarActivation&) = delete;
  AnyScalarActivation& operator=(const AnyScalarActivation&) = delete;

  ~AnyScalarActivation() = default;

  using CBase::clear;
  using CBase::emplace;
  using CBase::isDefined;
  using CBase::operator->;

  [[nodiscard]] nn::Scalar forward(nn::Scalar x) const {
    if (!isDefined()) {
      throw std::logic_error("AnyScalarActivation: empty");
    }
    return (*this)->forward(x);
  }

  [[nodiscard]] nn::Scalar derivative(nn::Scalar x) const {
    if (!isDefined()) {
      throw std::logic_error("AnyScalarActivation: empty");
    }
    return (*this)->derivative(x);
  }

  [[nodiscard]] nn::Scalar derivative_from_output(nn::Scalar y) const {
    if (!isDefined()) {
      throw std::logic_error("AnyScalarActivation: empty");
    }
    return (*this)->derivative_from_output(y);
  }
};

template<typename Activation>
struct ScalarActivationFn {
  [[nodiscard]] nn::Scalar forward(nn::Scalar x) const { return Activation::forward(x); }

  [[nodiscard]] nn::Scalar derivative(nn::Scalar x) const {
    if constexpr (requires { Activation::derivative(x); }) {
      return Activation::derivative(x);
    } else {
      const nn::Scalar y = Activation::forward(x);
      return Activation::derivative_from_output(y);
    }
  }

  [[nodiscard]] nn::Scalar derivative_from_output(nn::Scalar y) const {
    if constexpr (requires { Activation::derivative_from_output(y); }) {
      return Activation::derivative_from_output(y);
    } else {
      return Activation::derivative(y);
    }
  }
};

}  // namespace nn::activation
