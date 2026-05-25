#pragma once

#include "core/math/Linalg.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nn::train {

class DataLoader {
public:
  DataLoader(nn::MatrixXf X, nn::MatrixXf Y, std::size_t batch_size, bool shuffle = true)
      : X_(std::move(X)), Y_(std::move(Y)), batch_size_(batch_size), shuffle_(shuffle) {
    if (X_.rows() != Y_.rows()) {
      throw std::invalid_argument("DataLoader: X and Y batch dimension mismatch");
    }
    if (batch_size_ == 0) {
      throw std::invalid_argument("DataLoader: batch_size must be positive");
    }
    if (X_.rows() == 0) {
      throw std::invalid_argument("DataLoader: dataset is empty");
    }
    reset_indices();
  }

  void reset_epoch(std::uint64_t seed) {
    reset_indices();
    if (shuffle_) {
      std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
      std::shuffle(indices_.begin(), indices_.end(), rng);
    }
  }

  void reset_epoch() { reset_indices(); }

  [[nodiscard]] std::size_t num_batches() const {
    return (indices_.size() + batch_size_ - 1) / batch_size_;
  }

  [[nodiscard]] std::size_t dataset_size() const { return indices_.size(); }

  class Iterator {
  public:
    Iterator(const DataLoader* loader, std::size_t batch_index)
        : loader_(loader), batch_index_(batch_index) {}

    std::pair<nn::MatrixXf, nn::MatrixXf> operator*() const {
      return loader_->get_batch(batch_index_);
    }

    Iterator& operator++() {
      ++batch_index_;
      return *this;
    }

    bool operator!=(const Iterator& other) const {
      return batch_index_ != other.batch_index_ || loader_ != other.loader_;
    }

  private:
    const DataLoader* loader_;
    std::size_t batch_index_;
  };

  Iterator begin() const { return Iterator(this, 0); }

  Iterator end() const { return Iterator(this, num_batches()); }

private:
  void reset_indices() {
    indices_.resize(static_cast<std::size_t>(X_.rows()));
    for (std::size_t i = 0; i < indices_.size(); ++i) {
      indices_[i] = i;
    }
  }

  [[nodiscard]] std::pair<nn::MatrixXf, nn::MatrixXf> get_batch(std::size_t batch_index) const {
    const std::size_t start = batch_index * batch_size_;
    const std::size_t end = std::min(start + batch_size_, indices_.size());
    const std::size_t batch_rows = end - start;

    nn::MatrixXf batch_X(batch_rows, X_.cols());
    nn::MatrixXf batch_Y(batch_rows, Y_.cols());

    for (std::size_t i = 0; i < batch_rows; ++i) {
      const Eigen::Index row = static_cast<Eigen::Index>(indices_[start + i]);
      batch_X.row(static_cast<Eigen::Index>(i)) = X_.row(row);
      batch_Y.row(static_cast<Eigen::Index>(i)) = Y_.row(row);
    }
    return {std::move(batch_X), std::move(batch_Y)};
  }

  nn::MatrixXf X_;
  nn::MatrixXf Y_;
  std::size_t batch_size_;
  bool shuffle_;
  std::vector<std::size_t> indices_;
};

}  // namespace nn::train
