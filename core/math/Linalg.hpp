#pragma once

#include <Eigen/Dense>

namespace nn {

using Scalar = float;
using MatrixXf = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXf = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

struct In {
    Eigen::Index value;
};

struct Out {
    Eigen::Index value;
};

}  // namespace nn