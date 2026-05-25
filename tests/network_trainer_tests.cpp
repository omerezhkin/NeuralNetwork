#include <catch2/catch_test_macros.hpp>

#include "nn/loss/builtin.hpp"
#include "nn/model/network_builder.hpp"
#include "nn/optim/sgd.hpp"
#include "train/train.hpp"

TEST_CASE("XOR network trains", "[network][trainer]") {
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

  nn::loss::AnyLoss loss(nn::loss::Mse{});
  nn::optim::AnyOptimizer optimizer(nn::optim::Sgd{2.0F});
  nn::train::Trainer trainer(net, loss, optimizer);
  trainer.fit(X, Y, 3000, 3000);

  const nn::MatrixXf pred = net.predict(X);
  REQUIRE(pred(0, 0) < 0.35F);
  REQUIRE(pred(1, 0) > 0.65F);
  REQUIRE(pred(2, 0) > 0.65F);
  REQUIRE(pred(3, 0) < 0.35F);
}

TEST_CASE("DataLoader batching", "[dataloader]") {
  nn::MatrixXf X(5, 2);
  X.setRandom();
  nn::MatrixXf Y(5, 1);
  Y.setRandom();

  nn::train::DataLoader loader(std::move(X), std::move(Y), 2, false);
  REQUIRE(loader.num_batches() == 3);

  std::size_t total_rows = 0;
  for (auto it = loader.begin(); it != loader.end(); ++it) {
    auto [bx, by] = *it;
    REQUIRE(bx.rows() == by.rows());
    total_rows += static_cast<std::size_t>(bx.rows());
  }
  REQUIRE(total_rows == 5);
}
