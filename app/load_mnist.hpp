#pragma once

#include "core/math/Linalg.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace app {

inline std::uint32_t read_be32(std::ifstream& f) {
  std::uint8_t buf[4];
  f.read(reinterpret_cast<char*>(buf), 4);
  if (!f) {
    throw std::runtime_error("load_mnist: unexpected end of file");
  }
  return (static_cast<std::uint32_t>(buf[0]) << 24) |
         (static_cast<std::uint32_t>(buf[1]) << 16) |
         (static_cast<std::uint32_t>(buf[2]) << 8) |
         static_cast<std::uint32_t>(buf[3]);
}

inline nn::MatrixXf load_mnist_images(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("load_mnist_images: cannot open " + path);
  }

  const std::uint32_t magic = read_be32(f);
  if (magic != 0x00000803) {
    throw std::runtime_error("load_mnist_images: bad magic in " + path);
  }

  const auto n_images = static_cast<Eigen::Index>(read_be32(f));
  const auto n_rows   = static_cast<Eigen::Index>(read_be32(f));
  const auto n_cols   = static_cast<Eigen::Index>(read_be32(f));
  const Eigen::Index n_pixels = n_rows * n_cols;

  nn::MatrixXf X(n_images, n_pixels);

  for (Eigen::Index i = 0; i < n_images; ++i) {
    for (Eigen::Index j = 0; j < n_pixels; ++j) {
      std::uint8_t byte = 0;
      f.read(reinterpret_cast<char*>(&byte), 1);
      X(i, j) = static_cast<float>(byte) / 255.F;
    }
  }

  if (!f) {
    throw std::runtime_error("load_mnist_images: read error in " + path);
  }
  return X;
}

inline nn::MatrixXf load_mnist_labels_onehot(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("load_mnist_labels_onehot: cannot open " + path);
  }

  const std::uint32_t magic = read_be32(f);
  if (magic != 0x00000801) {
    throw std::runtime_error("load_mnist_labels_onehot: bad magic in " + path);
  }

  const auto n_labels = static_cast<Eigen::Index>(read_be32(f));
  nn::MatrixXf Y = nn::MatrixXf::Zero(n_labels, 10);

  for (Eigen::Index i = 0; i < n_labels; ++i) {
    std::uint8_t label = 0;
    f.read(reinterpret_cast<char*>(&label), 1);
    if (label > 9) {
      throw std::runtime_error("load_mnist_labels_onehot: label out of range");
    }
    Y(i, static_cast<Eigen::Index>(label)) = 1.F;
  }

  if (!f) {
    throw std::runtime_error("load_mnist_labels_onehot: read error in " + path);
  }
  return Y;
}

inline std::pair<nn::MatrixXf, nn::MatrixXf> load_mnist(const std::string& images_path,
                                                        const std::string& labels_path) {
  return {load_mnist_images(images_path), load_mnist_labels_onehot(labels_path)};
}

}  // namespace app
