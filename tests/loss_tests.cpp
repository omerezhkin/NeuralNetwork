#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nn/loss/mae.hpp"
#include "nn/loss/mse.hpp"

using Catch::Approx;

TEST_CASE("MSE value and gradient", "[loss]") {
  nn::loss::Mse mse;
  nn::MatrixXf pred(2, 1);
  nn::MatrixXf target(2, 1);
  pred << 1, 3;
  target << 0, 2;
  REQUIRE(mse.value(pred, target) == Approx(1.0F));
  const nn::MatrixXf grad = mse.backward(pred, target);
  REQUIRE(grad(0, 0) == Approx(1.0F));
  REQUIRE(grad(1, 0) == Approx(1.0F));
}

TEST_CASE("MAE gradient sign", "[loss]") {
  nn::loss::Mae mae;
  nn::MatrixXf pred(1, 1);
  nn::MatrixXf target(1, 1);
  pred << 2;
  target << 0;
  REQUIRE(mae.backward(pred, target)(0, 0) == Approx(1.F));
}
