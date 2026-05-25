#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nn/activation/builtin.hpp"
#include "nn/layers/activation_layer.hpp"

using Catch::Approx;

TEST_CASE("ReLU forward and derivative", "[activation]") {
  REQUIRE(nn::activation::ReLU::forward(-1.F) == Approx(0.F));
  REQUIRE(nn::activation::ReLU::forward(2.F) == Approx(2.F));
  REQUIRE(nn::activation::ReLU::derivative(-1.F) == Approx(0.F));
  REQUIRE(nn::activation::ReLU::derivative(2.F) == Approx(1.F));
}

TEST_CASE("Sigmoid layer backward shape", "[activation]") {
  nn::layers::ActivationLayer layer(nn::activation::Sigmoid{});
  nn::MatrixXf x(2, 2);
  x << -1, 0.5,
       0.2, 1.0;
  auto [y, cache] = layer.forward(x);
  nn::MatrixXf dLdY = nn::MatrixXf::Ones(2, 2);
  auto [dLdX, grads] = layer.backward(cache, dLdY);
  REQUIRE(dLdX.rows() == x.rows());
  REQUIRE(dLdX.cols() == x.cols());
  REQUIRE(grads);
}
