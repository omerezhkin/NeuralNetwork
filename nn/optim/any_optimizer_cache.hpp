#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace nn::optim {

class AnyOptimizerCache {
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
  AnyOptimizerCache() = default;

  template<typename T,
           typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, AnyOptimizerCache>>>
  explicit AnyOptimizerCache(T&& value)
      : impl_(std::make_unique<Model<std::remove_cvref_t<T>>>(std::forward<T>(value))) {}

  AnyOptimizerCache(const AnyOptimizerCache&) = delete;
  AnyOptimizerCache& operator=(const AnyOptimizerCache&) = delete;

  AnyOptimizerCache(AnyOptimizerCache&&) noexcept = default;
  AnyOptimizerCache& operator=(AnyOptimizerCache&&) noexcept = default;

  ~AnyOptimizerCache() = default;

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

  template<typename T>
  [[nodiscard]] T* as() noexcept {
    if (!impl_) {
      return nullptr;
    }
    using U = std::remove_cvref_t<T>;
    if (auto* m = dynamic_cast<Model<U>*>(impl_.get())) {
      return &m->value_;
    }
    return nullptr;
  }

  template<typename T>
  [[nodiscard]] const T* as() const noexcept {
    if (!impl_) {
      return nullptr;
    }
    using U = std::remove_cvref_t<T>;
    if (auto* m = dynamic_cast<const Model<U>*>(impl_.get())) {
      return &m->value_;
    }
    return nullptr;
  }
};

template<typename T>
[[nodiscard]] T* OptimizerCache_cast(AnyOptimizerCache* p) noexcept {
  return p ? p->as<T>() : nullptr;
}

template<typename T>
[[nodiscard]] const T* OptimizerCache_cast(const AnyOptimizerCache* p) noexcept {
  return p ? p->as<T>() : nullptr;
}

}  // namespace nn::optim
