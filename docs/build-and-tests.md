# Сборка и тесты

## Требования

- CMake 3.20+
- Компилятор с C++20 (Clang 14+, GCC 11+, MSVC 19.29+)
- Git (для FetchContent: Eigen, Catch2)

## Сборка

```bash
cmake -S . -B build -DNN_BUILD_EXAMPLE=ON -DNN_BUILD_TESTING=ON
cmake --build build
```

### Опции CMake

| Опция | По умолчанию | Описание |
|-------|--------------|----------|
| `NN_BUILD_EXAMPLE` | ON | Собрать `nn_example` из `app/main.cpp` |
| `NN_BUILD_TESTING` | ON | Собрать `nn_tests` и включить CTest |

## Запуск

```bash
./build/nn_example
ctest --test-dir build --output-on-failure
# или напрямую:
./build/tests/nn_tests
```

## Подключение как библиотеки

Цель `nn::nn` — INTERFACE, header-only. Include root — корень репозитория:

```cpp
#include "nn/nn.hpp"
```

## Preset (опционально)

```bash
cmake --preset default   # если добавлен CMakePresets.json
```
