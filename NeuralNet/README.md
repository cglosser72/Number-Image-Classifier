# Neural Network in C++

This project implements a simple **feedforward neural network** in C++ from scratch, using the **Eigen/Dense** library for matrix operations and the **ADAM optimization algorithm** for training. The network is designed to classify handwritten digits from the **MNIST dataset**.

## Features
- **Fully connected feedforward neural network**
- **ADAM optimizer** for efficient training
- Uses **Eigen/Dense** for matrix computations
- Trained on **MNIST dataset**
- Implemented in **Visual Studio** with **MinGW-w64 GCC**

---

## Network Architecture
The network is structured as follows:

- **Input Layer**: 784 nodes (28x28 pixel MNIST images, flattened)
- **Hidden Layer 1**: 128 neurons, **ReLU activation**
- **Hidden Layer 2**: 64 neurons, **ReLU activation**
- **Output Layer**: 10 neurons, **Softmax activation** (for classification)

### Forward Pass
1. Compute **z = Wx + b** for each layer.
2. Apply activation function (**ReLU or Softmax**).
3. Output the predicted probabilities for classification.

### Backpropagation
1. Compute loss using **categorical cross-entropy**.
2. Calculate gradients using the chain rule.
3. Update weights using **ADAM optimizer**.

---

## ADAM Optimization
ADAM (Adaptive Moment Estimation) is an advanced gradient descent method combining momentum and RMSprop. It updates parameters using:

- **Moving averages of past gradients** ($\vec m_t$)
    - $\vec m_t = \beta_1 \vec m_{t-1}+(1-\beta_1)*\vec \delta_t $

The idea here is to use exponential smoothing to cut down on noise in the loss in order to find global minimums rather than stochastic ones. 

- **Moving averages of squared gradients** ($\vec v_t$)
    - $\vec v_t = \beta_2 \vec v_{t-1}+(1-\beta_2)*\vec {\delta_t^2} $

We include a 2nd order term (the square of the gradients) in the smoothing. 

- **Bias correction terms**:
    -  $\hat{m}_t= \frac{\vec m_t}{(1-\beta_1^{t+1})} $
    -  $\hat{v}_t= \frac{\vec v_t}{(1-\beta_2^{t+1})} $

the reason that we do this is that if we don't, the algorithm will get stuck at the initialization point,  $\vec 0$.  The bias correction forces the system away from this point.

- The update rule for weights and biases:
    -  $\hat {v}'_i \equiv \sqrt{\hat{v}_i} $
    -  $\theta_{t+1} = \theta_t - \frac{\alpha}{\hat{v'}_t + \epsilon} \hat{m}_t $

where:
- $\alpha$ = learning rate
- $m_t$ = first moment estimate (mean of gradients)
- $v_t$ = second moment estimate (variance of gradients)
- $\epsilon$ = small constant for numerical stability

**Advantages of ADAM:**
- Fast convergence
- Works well with sparse gradients
- No need to tune learning rate manually

---

## Eigen/Dense Library
This project leverages **Eigen/Dense** for high-performance matrix computations. Eigen is a C++ template library that provides optimized numerical operations, crucial for neural network calculations.

### Why Eigen?
- **Fast**: Optimized for vectorized computations
- **Simple API**: Similar to NumPy (e.g., `MatrixXd A = MatrixXd::Random(3,3);`)
- **Header-only**: No need for external linking
- **Well-suited for ML/DL**

Example usage in this project:
```cpp
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
```

---

## How to Build and Run
### Prerequisites
- **Visual Studio**:
    -  Microsoft (R) C/C++ Optimizing Compiler Version 19.41.34120 for x86
- **Eigen library** (header-only, no installation required)
- **C++17 or later**

### Compilation
Use `g++` with Eigen:
```sh
g++ -std=c++17 -I /path_to_eigen neural_net.cpp -o neural_net.exe
```

### Running the Program
```sh
./neural_net.exe
```

---

## Author
Christopher Glosser, PhD


