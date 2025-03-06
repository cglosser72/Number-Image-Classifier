#include "neural_network.h"

using json = nlohmann::json;

Eigen::MatrixXd load_mnist_images(const std::string& filename, int num_images, int image_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file.ignore(16);
    Eigen::MatrixXd images(image_size, num_images);
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images(j, i) = (pixel / 127.5) - 1.0;
        }
    }
    file.close();
    return images;
}

// Function to read MNIST labels
Eigen::MatrixXd load_mnist_labels(const std::string& filename, int num_images, int num_classes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file.ignore(8); // Skip the header

    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(num_classes, num_images);
    for (int i = 0; i < num_images; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels(label, i) = 1.0; // One-hot encode labels
    }

    file.close();
    return labels;
}

// initialize from file

NeuralNetwork::NeuralNetwork(const std::string& filename) {

    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    std::cout << "Attempting to open: " << filename << std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open weight file: " + filename);
    }

    json weights_data;
    file >> weights_data;

    // Automatically detect the number of layers
    layer_sizes.clear();
    for (size_t i = 1; ; ++i) {
        std::string weight_key = "fc" + std::to_string(i) + ".weight";
        if (weights_data.contains(weight_key)) {
            int input_size = weights_data[weight_key][0].size();
            int output_size = weights_data[weight_key].size();
            if (layer_sizes.empty()) {
                layer_sizes.push_back(input_size);  // First layer's input size
            }
            layer_sizes.push_back(output_size);  // Next layer's output size
        }
        else {
            break;  // No more layers found
        }
    }

    std::cout << "Detected Network Structure: ";
    for (int size : layer_sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // Load Weights & Biases
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        // Load weights
        std::string weight_key = "fc" + std::to_string(i) + ".weight";
        Eigen::MatrixXd weight_matrix(layer_sizes[i], layer_sizes[i - 1]);

        auto weight_values = weights_data[weight_key];
        for (size_t row = 0; row < layer_sizes[i]; ++row) {
            for (size_t col = 0; col < layer_sizes[i - 1]; ++col) {
                weight_matrix(row, col) = weight_values[row][col];
            }
        }
        weights.push_back(weight_matrix);

        // Load biases
        std::string bias_key = "fc" + std::to_string(i) + ".bias";
        Eigen::MatrixXd bias_vector(layer_sizes[i], 1);

        auto bias_values = weights_data[bias_key];
        for (size_t row = 0; row < layer_sizes[i]; ++row) {
            bias_vector(row, 0) = bias_values[row];
        }
        biases.push_back(bias_vector);
    }
    initialize_adam();
    std::cout << "Loaded weights and biases from " << filename << std::endl;
}


NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        double fan_in = layer_sizes[i - 1];  // Input size
        double fan_out = layer_sizes[i];  // Output size
        double gain = sqrt(5);
        double bound = sqrt(6.0 / (fan_in + fan_out)*gain);  // PyTorch Kaiming Uniform formula
        double bound_bias = sqrt(1.0 / layer_sizes[i - 1]);  // PyTorch uses fan_in scaling
        std::uniform_real_distribution<double> dist(-bound, bound);
        std::uniform_real_distribution<double> bias_dist(-bound_bias, bound_bias);
        weights.emplace_back(Eigen::MatrixXd::NullaryExpr(layer_sizes[i], layer_sizes[i - 1], [&]() { return dist(gen); }));
        biases.emplace_back(Eigen::MatrixXd::NullaryExpr(layer_sizes[i], 1, [&]() { return bias_dist(gen); }));

    }
    initialize_adam();

}


void NeuralNetwork::initialize_adam() {
    for (size_t i = 0; i < weights.size(); ++i) {
        m_weights.push_back(Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
        v_weights.push_back(Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
        m_biases.push_back(Eigen::MatrixXd::Zero(biases[i].rows(), biases[i].cols()));
        v_biases.push_back(Eigen::MatrixXd::Zero(biases[i].rows(), biases[i].cols()));
    }
}

Eigen::MatrixXd NeuralNetwork::relu(const Eigen::MatrixXd& z) {
    return z.cwiseMax(0);
}


Eigen::MatrixXd NeuralNetwork::relu_derivative(const Eigen::MatrixXd& z) {
    return (z.array() > 0).cast<double>();
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd activation = input;
    activations.clear();
    activations.push_back(activation);

    for (size_t i = 0; i < weights.size() - 1; ++i) {
        activation = (weights[i] * activation) + biases[i].replicate(1, activation.cols());  //  Correct bias broadcasting
        activation = relu(activation);  //  Apply ReLU for hidden layers
        activations.push_back(activation);
    }

    //  Last layer (raw logits, no activation!)
    activation = (weights.back() * activation) + biases.back().replicate(1, activation.cols());  //  Correct bias broadcasting
    activations.push_back(activation);

    return activation;  //  Return raw logits, no ReLU/Softmax
}


Eigen::MatrixXd NeuralNetwork::softmax(const Eigen::MatrixXd& z) {
    Eigen::MatrixXd exp_z = (z.array() - z.maxCoeff()).exp();
    return exp_z.array().rowwise() / exp_z.array().colwise().sum();
}

double NeuralNetwork::cross_entropy_loss(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& logits) {
    Eigen::MatrixXd Y_pred = logits;
    Eigen::MatrixXd log_preds = Y_pred.array().log();
    double loss = -(Y_true.array() * log_preds.array()).sum() / Y_true.cols();
    return loss;
}


double accuracy(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred) {
    int correct = 0;
    for (int i = 0; i < Y_true.cols(); ++i) {
        Eigen::Index true_label, pred_label;
        Y_true.col(i).maxCoeff(&true_label);
        Y_pred.col(i).maxCoeff(&pred_label);
        if (true_label == pred_label) {
            correct++;
        }
    }
    return static_cast<double>(correct) / Y_true.cols();
}


void NeuralNetwork::train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, double learning_rate, int batch_size, bool use_adam) {
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    int time_step = 0;
    //double effective_lr = learning_rate * (batch_size / 256.0);  // Adjust LR based on batch size
    double effective_lr = learning_rate; 
    std::vector<Eigen::MatrixXd> batch_weight_gradients(weights.size());
    std::vector<Eigen::MatrixXd> batch_bias_gradients(biases.size());
    Eigen::MatrixXd Y_f = Y.cast<double>();
    int num_samples = X.cols();

    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, 2, ..., num_samples - 1

    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {

        std::shuffle(indices.begin(), indices.end(), g);

        // Apply shuffled order to X and Y
        Eigen::MatrixXd X_shuffled(X.rows(), X.cols());
        Eigen::MatrixXd Y_shuffled(Y_f.rows(), Y_f.cols());
        for (int i = 0; i < num_samples; ++i) {
            X_shuffled.col(i) = X.col(indices[i]);
            Y_shuffled.col(i) = Y_f.col(indices[i]);
        }

        double total_loss = 0.0;
        int num_batches = 0;
        // Initialize gradient accumulators
        for (size_t i = 0; i < weights.size(); ++i) {
            batch_weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
            batch_bias_gradients[i] = Eigen::MatrixXd::Zero(biases[i].rows(), biases[i].cols());
        }
        for (int batch_start = 0; batch_start < X.cols(); batch_start += batch_size) {
            int batch_end = std::min(batch_start + batch_size, static_cast<int>(X.cols()));

            Eigen::MatrixXd X_batch = X_shuffled.block(0, batch_start, X_shuffled.rows(), batch_end - batch_start);
            Eigen::MatrixXd Y_batch = Y_shuffled.block(0, batch_start, Y_shuffled.rows(), batch_end - batch_start);

            // Zero-out gradients before computing new gradients
            for (size_t i = 0; i < weights.size(); ++i) {
                batch_weight_gradients[i].setZero();
                batch_bias_gradients[i].setZero();
            }

            Eigen::MatrixXd logits = this->forward(X_batch);
            logits = this->softmax(logits);
            Eigen::MatrixXd error = logits - Y_batch;

            double batch_loss = cross_entropy_loss(Y_batch, logits);
            total_loss += batch_loss;
            num_batches++;

            if (time_step%100 == 0) {
                std::cout <<"time:"<< time_step << " Batch loss: " << batch_loss << std::endl;
            }

            // Backpropagate from last hidden layer to first hidden layer
            std::vector<Eigen::MatrixXd> delta(weights.size());
            delta.back() = error;

            for (int i = static_cast<int>(weights.size()) - 2; i >= 0; --i) {
                delta[i] = (weights[i + 1].transpose() * delta[i + 1]).cwiseProduct(relu_derivative(activations[i + 1]));
            }


            for (size_t i = 0; i < weights.size(); ++i) {
                batch_weight_gradients[i] += (delta[i] * activations[i].transpose()) / batch_size;
                batch_bias_gradients[i] += delta[i].rowwise().sum() / batch_size;
                double mean = batch_weight_gradients[i].mean();
                double stddev = std::sqrt((batch_weight_gradients[i].array() - mean).square().sum() / batch_weight_gradients[i].size());
                double mean_b = batch_bias_gradients[i].mean();
                double stddev_b = std::sqrt((batch_bias_gradients[i].array() - mean).square().sum() / batch_bias_gradients[i].size());
                if (use_adam) {
                    // Adam moment updates
                    // thus uses exponetial smoothing to update the weights vector
                    m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * batch_weight_gradients[i];
                    v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * batch_weight_gradients[i].array().square().matrix();
                    m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * batch_bias_gradients[i];
                    v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * batch_bias_gradients[i].array().square().matrix();

                    // Bias correction:  The vectors retain a "history" of pointing at 0 ("biased towards 0").
                    // Note that 1-beta is small, so the gradients can't drive the search away from 0.
                    // We fix this by amplifying them initially to remove this bias.
                    // after a few batches the bias correction -> 1
                    Eigen::MatrixXd m_hat_w = m_weights[i] / (1 - std::pow(beta1, time_step + 1));
                    Eigen::MatrixXd v_hat_w = (v_weights[i].array() / (1 - std::pow(beta2, time_step + 1))).array().sqrt();
                    Eigen::MatrixXd m_hat_b = m_biases[i] / (1 - std::pow(beta1, time_step + 1));
                    Eigen::MatrixXd v_hat_b = (v_biases[i].array() / (1 - std::pow(beta2, time_step + 1))).array().sqrt();

                    // Update weights using Adam formula
                    weights[i] -= effective_lr * (m_hat_w.array() / (v_hat_w.array().sqrt() + epsilon)).matrix();
                    biases[i] -= effective_lr * (m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix();
                }
                else {
                    // Standard SGD update
                    weights[i] -= effective_lr * batch_weight_gradients[i];
                    biases[i] -= effective_lr * batch_bias_gradients[i];
                }
            }
            time_step++;
        }
        std::cout << "Epoch " << (epoch + 1) << " - Loss: " << (total_loss / num_batches) << std::endl;
    }
}
