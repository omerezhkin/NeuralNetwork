#include <catch2/catch_test_macros.hpp>

#include "core/math/Linalg.hpp"

TEST_CASE("In/Out strong indices", "[core]") {
  const nn::In in{3};
  const nn::Out out{2};
  REQUIRE(in.value == 3);
  REQUIRE(out.value == 2);
}

TEST_CASE("Matrix layout B x features", "[core]") {
  nn::MatrixXf X(4, 2);
  X << 1, 2,
       3, 4,
       5, 6,
       7, 8;
  REQUIRE(X.rows() == 4);
  REQUIRE(X.cols() == 2);
}
