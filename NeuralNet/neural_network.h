#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>  // For std::shuffle
#include <random>     // For random device
#include <numeric>
#include <filesystem>  // C++17 feature
#include "include/json.hpp"  //  download this here:  https://github.com/nlohmann/json/tree/develop/include/nlohmann/json.hpp


class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes);
    NeuralNetwork(const std::string& filename);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, double learning_rate, int batch_size = 1, bool use_adam = false);

private:
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::MatrixXd> biases;
    std::vector<Eigen::MatrixXd> activations;
    std::vector<int> layer_sizes = {};

    Eigen::MatrixXd relu(const Eigen::MatrixXd& z);
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& z);
    Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& z);

    // Adam Optimizer Variables
    std::vector<Eigen::MatrixXd> m_weights;
    std::vector<Eigen::MatrixXd> v_weights;
    std::vector<Eigen::MatrixXd> m_biases;
    std::vector<Eigen::MatrixXd> v_biases;
    void initialize_adam();

    double cross_entropy_loss(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& logits);
};

// Function to read MNIST dataset
Eigen::MatrixXd load_mnist_images(const std::string& filename, int num_images, int image_size);
Eigen::MatrixXd load_mnist_labels(const std::string& filename, int num_images, int num_classes);
double accuracy(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred);

#endif // NEURAL_NETWORK_H
