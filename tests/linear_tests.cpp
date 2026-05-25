#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/random/random.hpp"
#include "nn/layers/any_gradients.hpp"
#include "nn/layers/linear.hpp"

TEST_CASE("Linear forward shape", "[linear]") {
  nn::random::Rng rng(0);
  nn::layers::Linear layer(nn::In{2}, nn::Out{3}, rng);
  nn::MatrixXf x(2, 2);
  x << 1, 0,
       0, 1;
  const nn::MatrixXf y = layer.predict(x);
  REQUIRE(y.rows() == 2);
  REQUIRE(y.cols() == 3);
}

TEST_CASE("Linear backward batch consistency", "[linear]") {
  nn::random::Rng rng(1);
  nn::layers::Linear layer(nn::In{2}, nn::Out{1}, rng);
  nn::MatrixXf x(3, 2);
  x << 1, 2,
       0, 1,
       2, 0;
  auto [y, cache] = layer.forward(x);
  nn::MatrixXf dLdY = nn::MatrixXf::Ones(y.rows(), y.cols());
  auto [dLdX, grads_any] = layer.backward(cache, dLdY);
  REQUIRE(dLdX.rows() == x.rows());
  REQUIRE(dLdX.cols() == x.cols());
  const auto* grads = nn::layers::AnyGradients_cast<nn::layers::Linear::Gradients>(&grads_any);
  REQUIRE(grads != nullptr);
  REQUIRE(grads->dW.rows() == 2);
  REQUIRE(grads->dW.cols() == 1);
  REQUIRE(grads->db.size() == 1);
}
