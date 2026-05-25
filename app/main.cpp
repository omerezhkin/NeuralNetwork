#include "nn/nn.hpp"

#include <iostream>
#include <string>

static float compute_accuracy(nn::model::Network& net, const nn::MatrixXf& X,
                               const nn::MatrixXf& Y_onehot) {
  const nn::MatrixXf pred = net.predict(X);
  int correct = 0;
  for (Eigen::Index i = 0; i < pred.rows(); ++i) {
    Eigen::Index pred_class = 0;
    pred.row(i).maxCoeff(&pred_class);

    Eigen::Index true_class = 0;
    Y_onehot.row(i).maxCoeff(&true_class);

    if (pred_class == true_class) {
      ++correct;
    }
  }
  return static_cast<float>(correct) / static_cast<float>(pred.rows());
}

int main(int argc, char** argv) {
  const std::string default_dir = "archive-4";
  const std::string data_dir = (argc >= 2) ? argv[1] : default_dir;

  const std::string train_images = data_dir + "/train-images.idx3-ubyte";
  const std::string train_labels = data_dir + "/train-labels.idx1-ubyte";
  const std::string test_images  = data_dir + "/t10k-images.idx3-ubyte";
  const std::string test_labels  = data_dir + "/t10k-labels.idx1-ubyte";

  std::cout << "Loading MNIST from: " << data_dir << '\n';

  auto [X_train, Y_train] = nn::io::load_mnist(train_images, train_labels);
  auto [X_test, Y_test] = nn::io::load_mnist(test_images, test_labels);

  std::cout << "Train: " << X_train.rows() << " x " << X_train.cols()
            << "  labels: " << Y_train.rows() << " x " << Y_train.cols() << '\n';
  std::cout << "Test:  " << X_test.rows() << " x " << X_test.cols()
            << "  labels: " << Y_test.rows() << " x " << Y_test.cols() << '\n';

  auto net = nn::model::NetworkBuilder(42)
                 .add_linear(nn::In{784}, nn::Out{256})
                 .add_relu()
                 .add_linear(nn::In{256}, nn::Out{10})
                 .build();

  nn::loss::AnyLoss loss(nn::loss::Mse{});
  nn::optim::AnyOptimizer optimizer(nn::optim::Adam{0.001F});

  nn::train::DataLoader loader(X_train, Y_train, 128, true);
  nn::train::Trainer trainer(net, loss, optimizer);
  trainer.set_data_loader(&loader);

  const std::size_t num_epochs = 10;
  std::cout << "\nTraining (" << num_epochs << " epochs, batch=128, Adam lr=0.001)\n";
  std::cout << "-----------------------------------------------------------\n";

  std::cout << "epoch  0  test_accuracy = " << compute_accuracy(net, X_test, Y_test) * 100.F
            << " %\n";

  for (std::size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    const float avg_loss = trainer.fit_epoch(static_cast<std::uint64_t>(epoch) * 1234ULL);
    const float acc = compute_accuracy(net, X_test, Y_test);
    std::cout << "epoch " << epoch << "  train_loss = " << avg_loss
              << "  test_accuracy = " << acc * 100.F << " %\n";
  }

  return 0;
}
