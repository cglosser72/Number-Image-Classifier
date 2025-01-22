## Project: Handwritten Digit Classification with PyTorch

This project trains and evaluates neural networks to classify handwritten digits from the MNIST dataset. It includes various functionalities such as dynamic activation function selection, training visualization, and evaluation metrics.

---

## Contents
1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)
4. [Functions and Features](#functions-and-features)
5. [Model Saving and Loading](#model-saving-and-loading)
6. [Evaluation](#evaluation)

---

## Overview
This project demonstrates the use of PyTorch to build and train neural networks for digit classification. It includes utilities for:
- Visualizing training and validation loss.
- Using dynamic activation functions.
- Evaluating model performance.
- Visualizing mislabeled examples and activation functions.

The models used are simple feedforward neural networks, including:
1. `SimpleNN`: A basic fully connected network.
2. `BlendedNN`: A network with different activation functions in different layers.

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- seaborn
- scikit-learn
- torchviz
- networkx
- graphviz

### Installation
Install the required dependencies:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn torchviz networkx graphviz
```

---

## Usage

### Training the Model
To train a model, use the `train_model` or `train_and_validate` function:
```python
from ic import SimpleNN, train_model

# Define the model, criterion, optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, optimizer, criterion, train_loader, epochs=5)
```

### Evaluating Accuracy
To calculate accuracy:
```python
from ic import evaluate_accuracy

accuracy = evaluate_accuracy(model, test_loader, device)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Plotting Mislabeled Images
Visualize examples of mislabeled images:
```python
from ic import plot_mislabeled_images

plot_mislabeled_images(model, test_loader, device, num_examples=10)
```

### Saving and Loading the Model
To save the model:
```python
torch.save(model.state_dict(), "model.pth")
```

To load the model:
```python
from ic import SimpleNN

model = SimpleNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

## Functions and Features

### Visualization
- **`plot_activation`**: Visualizes activation functions (e.g., ReLU, Sigmoid).
- **`draw_neural_network`**: Generates a schematic representation of the network architecture.

### Training Utilities
- **`train_model`**: Basic training loop.
- **`train_and_validate`**: Tracks training and validation loss over epochs.

### Evaluation Utilities
- **`evaluate_model`**: Generates predictions for the test dataset.
- **`evaluate_accuracy`**: Calculates overall accuracy.
- **`plot_mislabeled_images`**: Displays examples where the model made incorrect predictions.

---

## Model Saving and Loading

### Saving
Save model parameters:
```python
torch.save(model.state_dict(), "model.pth")
```

### Loading
Load parameters into a model:
```python
model = SimpleNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

## Evaluation
- **Precision, Recall, and F-Score**:
  Use the `calculate_metrics` function:
  ```python
  from sklearn.metrics import precision_score, recall_score

  precision = precision_score(true_labels, predicted_labels, average='weighted')
  recall = recall_score(true_labels, predicted_labels, average='weighted')
  ```

- **Confusion Matrix**:
  ```python
  from sklearn.metrics import confusion_matrix
  import seaborn as sns

  cm = confusion_matrix(true_labels, predicted_labels)
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  plt.show()
  ```

