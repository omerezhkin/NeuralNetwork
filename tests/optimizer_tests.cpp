#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nn/optim/adam.hpp"
#include "nn/optim/sgd.hpp"

using Catch::Approx;

TEST_CASE("SGD updates matrix", "[optimizer]") {
  nn::optim::Sgd sgd(0.1F);
  nn::MatrixXf param(1, 1);
  param << 1.F;
  nn::MatrixXf grad(1, 1);
  grad << 2.F;
  nn::optim::AnyOptimizerCache cache;
  sgd.update_weights(param, grad, cache);
  REQUIRE(param(0, 0) == Approx(0.8F));
}

TEST_CASE("Adam one step changes parameters", "[optimizer]") {
  nn::optim::Adam adam(0.01F);
  nn::MatrixXf param(2, 1);
  param << 0.5F, -0.5F;
  nn::MatrixXf grad(2, 1);
  grad << 0.1F, -0.1F;
  nn::optim::AnyOptimizerCache cache;
  adam.update_weights(param, grad, cache);
  adam.iter_step();
  REQUIRE(param(0, 0) != Approx(0.5F));
}
