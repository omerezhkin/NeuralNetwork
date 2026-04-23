#include "core/math/Linalg.hpp"
#include "nn/loss/mse.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/sgd.hpp"
#include "train/train.hpp"

#include <iostream>

int main() {
  nn::MatrixXf X(4, 2);
  X << 0, 0,
       0, 1,
       1, 0,
       1, 1;

  nn::MatrixXf Y(4, 1);
  Y << 0,
       1,
       1,
       0;

  auto net = nn::model::NetworkBuilder(42)
      .add_linear(nn::In{2}, nn::Out{4})
      .add_sigmoid()
      .add_linear(nn::In{4}, nn::Out{1})
      .add_sigmoid()
      .build();

  nn::loss::Mse loss;
  nn::optim::Sgd sgd(1.0F);

  nn::train::Trainer trainer(net, loss, sgd);
  trainer.fit(X, Y, 5000, 500);

  nn::MatrixXf result = net.predict(X);
  std::cout << "\nPredictions after training:\n" << result << '\n';

  return 0;
}
