#pragma once

#include <cstddef>
#include <memory>
#include <utility>

namespace nn::library {

//---------------------------------------------------------------------------
// How To Use
//---------------------------------------------------------------------------
//
// CAnyMovable хранит только move-конструируемый объект, без SOO: перемещение дешёвое.
// operator->() без проверок; перед доступом вызывайте isDefined().
//
// 1) Интерфейс:
//    template<class TBase>
//    class IAny : public TBase {
//    public:
//      virtual void print() const = 0;
//    };
//
// 2) Реализация:
//    template<class TBase, class TObject>
//    class CAnyImpl : public TBase {
//      using CBase = TBase;
//    public:
//      using CBase::CBase;
//      void print() const override { ... CBase::Object() ... }
//    };
//
// 3) Обертка:
//    class CAny : public CAnyMovable<IAny, CAnyImpl> {
//      using CBase = CAnyMovable<IAny, CAnyImpl>;
//    public:
//      using CBase::CBase;
//    };
//
//---------------------------------------------------------------------------

template<template<class> class TInterface, template<class, class> class TImplementation>
class CAnyMovable {
  class IObjectStored;
  using CStoredPtr = std::unique_ptr<IObjectStored>;
  template<class T>
  using CObjType = std::remove_reference_t<T>;

public:
  CAnyMovable() = default;

  template<class T>
  CAnyMovable(T&& Object)
      : pIObject_(std::make_unique<CObjectStored<CObjType<T>>>(std::forward<T>(Object))) {}

  template<class T, class... TArgs>
  CAnyMovable(std::in_place_type_t<T>, TArgs&&... args)
      : pIObject_(std::make_unique<CObjectStored<T>>(std::forward<TArgs>(args)...)) {}

  CAnyMovable(const CAnyMovable& Other) = delete;

  CAnyMovable(CAnyMovable&& Other) noexcept = default;

  CAnyMovable& operator=(const CAnyMovable& Other) = delete;

  CAnyMovable& operator=(CAnyMovable&& Other) noexcept = default;

  [[nodiscard]] bool isDefined() const { return static_cast<bool>(pIObject_); }

  /// База с виртуальным деструктором, чтобы пользователю не объявлять его в интерфейсе.
  class IEmpty {
  protected:
    virtual ~IEmpty() = default;
  };

  TInterface<IEmpty>* operator->() { return pIObject_.get(); }

  const TInterface<IEmpty>* operator->() const { return pIObject_.get(); }

  template<class TObject, class... TArgs>
  void emplace(TArgs&&... args) {
    pIObject_ =
        std::make_unique<CObjectStored<TObject>>(std::forward<TArgs>(args)...);
  }

  void clear() { pIObject_.reset(); }

protected:
  ~CAnyMovable() = default;

private:
  class IObjectStored : public TInterface<IEmpty> {
  public:
    ~IObjectStored() override = default;
  };

  template<class TObject>
  class CObjectKeeper : public IObjectStored {
  public:
    using CObjectType = TObject;

    CObjectKeeper(TObject&& Object) : Object_(std::move(Object)) {}

    ~CObjectKeeper() override = default;

    template<class... TArgs>
    explicit CObjectKeeper(TArgs&&... args) : Object_(std::forward<TArgs>(args)...) {}

  protected:
    TObject& Object() { return Object_; }

    const TObject& Object() const { return Object_; }

  private:
    TObject Object_;
  };

  template<class T, std::size_t Tsize>
  class CObjectKeeper<T[Tsize]> : public IObjectStored {
  public:
    using CBaseType = T;
    using CObjectType = CBaseType[Tsize];

    CObjectKeeper(CObjectType&& Object) noexcept
        : CObjectKeeper(std::move(Object), std::make_integer_sequence<std::size_t, Tsize>{}) {}

    ~CObjectKeeper() override = default;

  protected:
    CObjectType& Object() { return Object_; }

    const CObjectType& Object() const { return Object_; }

  private:
    template<std::size_t... TIndex>
    CObjectKeeper(CObjectType&& Object, std::integer_sequence<std::size_t, TIndex...>) noexcept
        : Object_{std::move(Object[TIndex])...} {}

    CObjectType Object_;
  };

  template<class TObject>
  class CObjectStored : public TImplementation<CObjectKeeper<TObject>, TObject> {
    using CBase = TImplementation<CObjectKeeper<TObject>, TObject>;

  public:
    using CBase::CBase;

    ~CObjectStored() override = default;
  };

protected:
  CStoredPtr& StoredPtr() { return pIObject_; }

  const CStoredPtr& StoredPtr() const { return pIObject_; }

private:
  CStoredPtr pIObject_;
};

}  // namespace nn::library
