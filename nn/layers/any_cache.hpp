#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace nn::layers {

class AnyCache {
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
  AnyCache() = default;

  template<typename T,
           typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, AnyCache>>>
  explicit AnyCache(T&& value)
      : impl_(std::make_unique<Model<std::remove_cvref_t<T>>>(std::forward<T>(value))) {}

  AnyCache(const AnyCache&) = delete;
  AnyCache& operator=(const AnyCache&) = delete;

  AnyCache(AnyCache&&) noexcept = default;
  AnyCache& operator=(AnyCache&&) noexcept = default;

  ~AnyCache() = default;

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

  template<typename T>
  friend const std::remove_cvref_t<T>* AnyCache_cast(const AnyCache* p) noexcept {
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
  friend std::remove_cvref_t<T>* AnyCache_cast(AnyCache* p) noexcept {
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
