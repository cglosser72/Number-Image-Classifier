import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchviz import make_dot
from graphviz import Digraph
from IPython.display import Image, display
import matplotlib.pyplot as plt
import networkx as nx
import os
import ic
import numpy as np

def print_examples():
    # Define the transformation to convert images to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the MNIST training dataset
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Select a few random examples from the dataset
    num_samples = 5
    indices = torch.randint(0, len(mnist_data), (num_samples,))

    # Plot the images
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 6))
    for i, idx in enumerate(indices):
        image, label = mnist_data[idx]
        axes[i].imshow(image.squeeze(), cmap='gray')  # Remove channel dimension and show the image
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')  # Hide axis

    plt.show()

def plot_activation(x, relu = torch.nn.ReLU(), lbl = "ReLU"):
    # Apply the ReLU function
    
    y = relu(x)

    # Plot the ReLU function
    plt.figure(figsize=(8, 6))
    plt.plot(x.numpy(), y.numpy(), label=lbl+"(x)", color="k", linewidth=2)
    plt.axhline(0, color='black', linewidth=0.5, linestyle="--")  # Add x-axis
    plt.axvline(0, color='black', linewidth=0.5, linestyle="--")  # Add y-axis
    plt.title(lbl+" Activation Function", fontsize=16, color="#b03b5a")
    plt.xlabel("Input (x)", fontsize=12)
    plt.ylabel("Output ("+lbl+"(x))", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.show()

class SimpleNN(nn.Module):
    def __init__(self, activation_function=nn.ReLU):
        """
        Initialize the neural network with a selectable activation function.

        Args:
            activation_function (nn.Module): The activation function to use. 
                                             Defaults to nn.ReLU.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28) to hidden layer
        self.fc2 = nn.Linear(128, 64)       # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 10)        # Hidden layer to output layer (10 classes)
        self.activation = activation_function()  # Dynamically set activation function

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
#        x = self.activation(self.fc1(x))
#        x = self.activation(self.fc2(x))        
#        x = self.fc3(x)          # Output layer (no activation, logits)
        self.activations_fc1 = self.activation(self.fc1(x))
        self.activations_fc2 = self.activation(self.fc2(self.activations_fc1))
        x = self.fc3(self.activations_fc2)        
        return x

class BlendedNN(nn.Module):
    def __init__(self):
        """
        Initialize the neural network with a selectable activation function.

        Args:
            activation_function (nn.Module): The activation function to use. 
                                             Defaults to nn.ReLU.
        """
        super(BlendedNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28) to hidden layer
        self.fc2 = nn.Linear(128, 64)       # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 10)        # Hidden layer to output layer (10 classes)
        self.activation1 = nn.ReLU()        # First activation function
        self.activation2 = nn.LeakyReLU(negative_slope=0.01)  # Second activation function
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
#        x = self.activation(self.fc1(x))
#        x = self.activation(self.fc2(x))        
#        x = self.fc3(x)          # Output layer (no activation, logits)
        self.activations_fc1 = self.activation1(self.fc1(x))
        self.activations_fc2 = self.activation2(self.fc2(self.activations_fc1))
        x = self.fc3(self.activations_fc2)        
        return x

def train_model(model, optimizer, criterion, train_loader, epochs=5):
    print(f"epochs to run: {epochs}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
        
            # Backward pass
            loss.backward()
        
            # Update weights
            optimizer.step()
        
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


def draw_neural_network(layer_sizes):
    """
    Draws a neural network diagram with nodes and connections.
    Args:
        layer_sizes (list): List of integers, where each integer is the number of nodes in a layer.
    """
    G = nx.DiGraph()
    pos = {}  # Dictionary to store positions of nodes for plotting
    node_count = 0

    # Add nodes and positions for each layer
    for i, size in enumerate(layer_sizes):
        for j in range(size):
            node_name = f'L{i}_N{j}'
            G.add_node(node_name, layer=i)
            pos[node_name] = (i, -j)  # (layer index, negative for top-down visualization)
            node_count += 1

    # Add edges between layers
    for i in range(len(layer_sizes) - 1):
        for src in range(layer_sizes[i]):
            for dst in range(layer_sizes[i + 1]):
                G.add_edge(f'L{i}_N{src}', f'L{i + 1}_N{dst}')

    # Draw the graph
    plt.figure(figsize=(20, 30))
    nx.draw(
        G, pos, with_labels=False, node_size=500, node_color='lightblue',
        edge_color='gray', arrows=False
    )

    # Label nodes by layer
    for node, (x, y) in pos.items():
        layer, neuron = node.split('_')
        label = f'{neuron[1:]}'  # Just show the neuron index
        plt.text(x, y, label, fontsize=8, ha='center', va='center', color='black')

    # Label layers
    for i, size in enumerate(layer_sizes):
        plt.text(i, 1, f'Layer {i}', fontsize=12, ha='center', color='darkred')

    plt.title('Neural Network Visualization')
    plt.axis('off')
    plt.show()

# Function to evaluate the model and get predictions
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()    
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Send images to device (GPU or CPU)
            images = images.to(device)  # e.g., device = 'cuda' or 'cpu'
            labels = labels.to(device)

            # Get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Store the true labels and predicted labels
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return true_labels, predicted_labels




# Function to evaluate and visualize mislabeled examples
def plot_mislabeled_images(model, test_loader, device, num_examples=10):
    """
    Evaluate the model and plot examples of mislabeled images.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device (CPU or GPU) to use for computation.
        num_examples (int): Number of mislabeled examples to display.
    """
    model.to(device)
    model.eval()
    true_labels = []
    predicted_labels = []
    images_list = []

    # Perform evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            images_list.extend(images.cpu().numpy())

    # Find indices of mislabeled examples
    mismatched_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) if true != pred]

    # Plot the mislabeled examples
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(mismatched_indices[:num_examples]):
        image = images_list[idx].squeeze()  # Remove extra dimensions
        true_label = true_labels[idx]
        predicted_label = predicted_labels[idx]

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

        # Compute average losses
        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total
