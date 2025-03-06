#include "neural_network.h"
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iostream>

int main() {
    std::random_device rd;
    std::mt19937 gen(42);  // Use fixed seed

    std::vector<int> layer_sizes = { 784, 128, 64, 10 };  // initialize layers 
    NeuralNetwork nn(layer_sizes);
 //   NeuralNetwork nn("weights.json"); // use file def to initialize layers

    std::string train_images_path = "C:\\Users\\cglos\\Number-Image-Classifier\\data\\MNIST\\raw\\train-images-idx3-ubyte";
    std::string train_labels_path = "C:\\Users\\cglos\\Number-Image-Classifier\\data\\MNIST\\raw\\train-labels-idx1-ubyte";

    std::string test_images_path = "C:\\Users\\cglos\\Number-Image-Classifier\\data\\MNIST\\raw\\t10k-images-idx3-ubyte";
    std::string test_labels_path = "C:\\Users\\cglos\\Number-Image-Classifier\\data\\MNIST\\raw\\t10k-labels-idx1-ubyte";

    int num_train_images = 60000; // 60000 max
    int num_test_images = 10000;  // 10000 max
    int image_size = 28 * 28;
    int num_classes = 10;
    // std::string input = "";
    
    std::cout << "Load MNIST train data" << std::endl;
    Eigen::MatrixXd X_train = load_mnist_images(train_images_path, num_train_images, image_size);
    std::cout << "Load MNIST test data" << std::endl;
    Eigen::MatrixXd X_test = load_mnist_images(test_images_path, num_test_images, image_size);
    std::cout << "Load MNIST train labels" << std::endl;
    Eigen::MatrixXd Y_train = load_mnist_labels(train_labels_path, num_train_images, num_classes);
    std::cout << "Load MNIST test labels" << std::endl;
    Eigen::MatrixXd Y_test = load_mnist_labels(test_labels_path, num_test_images, num_classes);

    std::cout << "Begin" << std::endl;
    nn.train(X_train, Y_train, 5, 0.001, 64, true);

    Eigen::MatrixXd predictions_train = nn.forward(X_train);
    Eigen::MatrixXd predictions_test = nn.forward(X_test);
    std::cout << "Train Accuracy: " << accuracy(Y_train, predictions_train) << std::endl;
    std::cout << "Test Accuracy: " << accuracy(Y_test, predictions_test) << std::endl;

    std::cout << "Exiting: ";
    return 0;
}
