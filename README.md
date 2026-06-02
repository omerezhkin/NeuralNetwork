# NeuralNetworkCpp

Header-only C++20 библиотека полносвязных нейросетей на Eigen 3.4 (курсовая работа).

## Конвенция размерностей

- **Строки матрицы = элементы батча (B)**
- **Столбцы = признаки (in) или выходы (out)**
- Линейный слой: `Y = X * W + b` (broadcast bias по строкам)

## Быстрый старт

```cmake
add_subdirectory(path/to/NeuralNetworkCpp)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE nn::nn)
```

```cpp
#include "nn/nn.hpp"

auto net = nn::model::NetworkBuilder(42)
    .add_linear(nn::In{2}, nn::Out{4})
    .add_relu()
    .add_linear(nn::In{4}, nn::Out{1})
    .add_sigmoid()
    .build();

nn::loss::AnyLoss loss(nn::loss::Mse{});
nn::optim::AnyOptimizer opt(nn::optim::Sgd{0.01F});
nn::train::Trainer trainer(net, loss, opt);
```

## Структура

| Каталог | Назначение |
|---------|------------|
| `core/` | `CAnyMovable`, `Linalg`, `random` |
| `nn/` | слои, loss, optim, model, activation |
| `train/` | `Trainer`, `DataLoader` |
| `app/` | пример обучения на MNIST + `load_mnist.hpp` |
| `tests/` | unit-тесты (Catch2) |

