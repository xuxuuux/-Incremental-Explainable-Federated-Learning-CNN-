'''
This code file is used to train the Spiking Neural Network (SNN) model
on the SVHN dataset and visualize the performance using a confusion matrix.
The model is based on the SNN architecture, which processes input images as spike sequences.

In this project, we use the **PoissonEncoder** to convert static images into spike sequences.
The training and evaluation will include the calculation of confusion matrix and F1 scores.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, encoding
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from models.SNN import SNNNet

# ===============================
# Step 1: Data Preprocessing
# ===============================
# Data transformation pipeline:
# 1. Resize all images to 32x32 pixels to match the SNN input size.
# 2. Convert images to PyTorch tensors.
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32, as expected by the SNN model
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Load the SVHN dataset
train_dataset = torchvision.datasets.SVHN(
    root='../data/data_SVHN', split='train', download=False, transform=transform
)
test_dataset = torchvision.datasets.SVHN(
    root='../data/data_SVHN', split='test', download=False, transform=transform
)

# Create DataLoader for training and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Step 2: Train and Evaluate Model
# ===============================
def train_and_evaluate(epochs=5, T=8):  # T is the simulation time steps (number of spikes)
    # Initialize the SNN model
    net = SNNNet().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for classification
    optimizer = optim.Adam(net.parameters(), lr=0.005)  # Adam optimizer

    # Initialize the Poisson encoder for spike encoding
    encoder = encoding.PoissonEncoder()

    # Lists to store true labels and predicted labels for confusion matrix
    y_true = []
    y_pred = []

    # Training loop
    for epoch in range(epochs):
        net.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate through the training dataset
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Convert static images to spike sequences: shape [T, batch, C, H, W]
            x_seq = torch.stack([encoder(inputs) for _ in range(T)])

            optimizer.zero_grad()  # Reset gradients before each batch

            functional.reset_net(net)  # Reset membrane potential at the start of each batch

            # Forward pass
            outputs = net(x_seq)
            loss = criterion(outputs, labels)  # Calculate loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # Print every 100 iterations
                print(f"[Epoch {epoch+1}, Iter {i+1}] loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        # Evaluation after each epoch
        net.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No gradients needed for evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                x_seq = torch.stack([encoder(inputs) for _ in range(T)])
                functional.reset_net(net)

                # Forward pass
                outputs = net(x_seq)
                _, predicted = outputs.max(1)  # Get the predicted labels
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Record true and predicted labels for confusion matrix
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        print(f"Epoch {epoch+1}: Test Accuracy = {100. * correct / total:.2f}%")

    # Save the trained model
    torch.save(net.state_dict(), "snn_svhn.pth")
    print("Model saved to snn_svhn.pth")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix

    # Calculate F1 score
    f1_scores = f1_score(y_true, y_pred, average=None)

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title('Normalized Confusion Matrix (SNN)', fontsize=16)
    plt.show()

    # Print F1 scores for each class and the average F1 score
    print(f"F1 Scores for each class: {f1_scores}")
    print(f"Average F1 Score: {np.mean(f1_scores)}")


if __name__ == "__main__":
    # Train and evaluate the SNN model with 20 epochs and 50 time steps for spikes
    train_and_evaluate(epochs=20, T=50)
