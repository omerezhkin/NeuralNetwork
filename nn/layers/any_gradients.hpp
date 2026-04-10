#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace nn::layers {

/// Стирание типа структуры градиентов слоя. Сами сигналы в backprop — \c nn::MatrixXf в \c any_layer.hpp.
class AnyGradients {
  struct Concept {
    virtual ~Concept() = default;
  };

  template<typename T>
  struct Model final : Concept {
    T value_;

    template<typename U>
    explicit Model(U&& u) : value_(std::forward<U>(u)) {}
  };

  std::unique_ptr<Concept> impl_;

public:
  AnyGradients() = default;

  template<typename T,
           typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, AnyGradients>>>
  explicit AnyGradients(T&& value)
      : impl_(std::make_unique<Model<std::remove_cvref_t<T>>>(std::forward<T>(value))) {}

  AnyGradients(const AnyGradients&) = delete;
  AnyGradients& operator=(const AnyGradients&) = delete;

  AnyGradients(AnyGradients&&) noexcept = default;
  AnyGradients& operator=(AnyGradients&&) noexcept = default;

  ~AnyGradients() = default;

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

  template<typename T>
  friend const std::remove_cvref_t<T>* AnyGradients_cast(const AnyGradients* p) noexcept {
    if (!p || !p->impl_) {
      return nullptr;
    }
    using U = std::remove_cvref_t<T>;
    if (auto* m = dynamic_cast<const Model<U>*>(p->impl_.get())) {
      return &m->value_;
    }
    return nullptr;
  }

  template<typename T>
  friend std::remove_cvref_t<T>* AnyGradients_cast(AnyGradients* p) noexcept {
    if (!p || !p->impl_) {
      return nullptr;
    }
    using U = std::remove_cvref_t<T>;
    if (auto* m = dynamic_cast<Model<U>*>(p->impl_.get())) {
      return &m->value_;
    }
    return nullptr;
  }
};

}  // namespace nn::layers
