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

- **Moving averages of past gradients** ($m_t$)
- **Moving averages of squared gradients** ($v_t$)
- **Bias correction terms**

The update rule for weights:

\[ \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t \]

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
#include <Eigen/Dense>
using namespace Eigen;

MatrixXd weights = MatrixXd::Random(128, 784);
VectorXd input = VectorXd::Random(784);
VectorXd output = weights * input;
```

---

## How to Build and Run
### Prerequisites
- **Visual Studio Code** with **MinGW-w64 (GCC 14.2.0)**
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

## Future Improvements
- Implement **batch normalization**
- Add support for **dropout**
- Expand to **CNNs for better image classification**
- Optimize matrix operations with **BLAS/LAPACK**

---

## Author
[Your Name]

---

## License
MIT License

