'''
This code file is used for federated learning, where the central model
is trained using average aggregation (FedAvg). You can train with either
IID or Non-IID data distributions. By default, the code uses the Non-IID setting.
The training dataset is the SVHN dataset, which is split across multiple clients.

The model used in this federated learning setup is a Spiking Neural Network (SNN).
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
import os
from collections import OrderedDict
from models.SNN import SNNNet  # Ensure you have the SNNNet class in the 'models/SNN.py'
from spikingjelly.activation_based import encoding, functional

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the SVHN dataset
def load_svhn():
    '''
    Load the SVHN dataset and preprocess the images.
    The images are resized to 32x32 pixels and converted to tensors.
    '''
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])
    trainset = datasets.SVHN(root="../data/data_SVHN", split='train', download=False, transform=transform)
    testset = datasets.SVHN(root="../data/data_SVHN", split='test', download=False, transform=transform)
    return trainset, testset


# Split data into IID or Non-IID client datasets
def split_data(dataset, num_clients, iid, alpha=1.0, seed=42):
    '''
    Split the dataset into IID or Non-IID client datasets.
    If IID, data is shuffled randomly. If Non-IID, data is split using a Dirichlet distribution.

    Args:
        dataset: Dataset containing images and labels.
        num_clients: Number of clients in federated learning.
        iid: Whether to use IID or Non-IID data distribution.
        alpha: Dirichlet distribution parameter controlling data heterogeneity.
        seed: Random seed for reproducibility.

    Returns:
        A list of sample indices for each client.
    '''
    indices = list(range(len(dataset)))
    if iid:
        random.shuffle(indices)
        return [indices[i::num_clients] for i in range(num_clients)]
    else:
        np.random.seed(seed)
        random.seed(seed)

        labels = np.array(dataset.labels) if hasattr(dataset, 'labels') else np.array(dataset.targets)
        n_classes = len(np.unique(labels))
        class_indices = [np.where(labels == c)[0] for c in range(n_classes)]  # Group indices by class
        distribution = np.random.dirichlet([alpha] * num_clients, n_classes)  # Dirichlet distribution

        client_indices = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            class_idcs = class_indices[c]
            np.random.shuffle(class_idcs)
            proportions = distribution[c]
            split_points = (np.cumsum(proportions) * len(class_idcs)).astype(int)[:-1]
            client_splits = np.split(class_idcs, split_points)
            for client_id in range(num_clients):
                client_indices[client_id].extend(client_splits[client_id].tolist())

        for client_id in range(num_clients):
            np.random.shuffle(client_indices[client_id])

        return client_indices


# Train a local client model
def train_client(model, train_loader, epochs, lr, T):
    '''
    Train a local client model using its own dataset.

    Args:
        model: The local model to be trained.
        train_loader: DataLoader for the client's local training data.
        epochs: Number of training epochs for the client.
        lr: Learning rate for the optimizer.
        T: Number of time steps (spike sequences).

    Returns:
        The updated model parameters, average loss, and training accuracy.
    '''
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    encoder = encoding.PoissonEncoder()  # Use Poisson encoder for spike sequences

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Convert static images to spike sequences [T, batch, C, H, W]
            x_seq = torch.stack([encoder(images) for _ in range(T)])

            optimizer.zero_grad()
            functional.reset_net(model)  # Reset the membrane potential after each batch

            # Forward pass
            outputs = model(x_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return model.state_dict(), avg_loss, accuracy


# Federated Averaging (FedAvg) to aggregate local models into a global model
def aggregate_models(global_model, client_models):
    '''
    Aggregate the local client models using the Federated Averaging method (FedAvg).

    Args:
        global_model: The global model to be updated.
        client_models: A list of client model state_dicts.
    '''
    global_state_dict = OrderedDict()
    for key in global_model.state_dict().keys():
        global_state_dict[key] = torch.mean(
            torch.stack([client_models[i][key] for i in range(len(client_models))]), dim=0
        )
    global_model.load_state_dict(global_state_dict)


# Test the global model
def test_model(model, test_loader, T):
    '''
    Evaluate the global model on the test dataset.

    Args:
        model: The model to be evaluated.
        test_loader: DataLoader for the test dataset.
        T: Number of time steps (spike sequences).

    Returns:
        The average loss and accuracy on the test dataset.
    '''
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    encoder = encoding.PoissonEncoder()  # Use Poisson encoder for spike sequences

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Convert static images to spike sequences
            x_seq = torch.stack([encoder(images) for _ in range(T)])

            functional.reset_net(model)

            outputs = model(x_seq)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Federated Learning settings
num_clients = 5  # Number of client models in the federated system
iid = False  # Whether to use IID or Non-IID data distribution
num_rounds = 30  # Number of federated training rounds
local_epochs = 20  # Number of local epochs per federated round
batch_size = 64
lr = 0.007  # Learning rate
T = 50  # Number of time steps (spike sequence length)

# Load and split the SVHN dataset into IID/Non-IID
trainset, testset = load_svhn()
client_indices = split_data(trainset, num_clients, iid, alpha=1.0, seed=42)

# Initialize results file and remove if it exists
results_file = "30epoch_federated_results_snn_noniid_5.txt"
if os.path.exists(results_file):
    os.remove(results_file)

# Prepare the DataLoaders for clients and test set
train_loaders = [
    data.DataLoader(
        data.Subset(trainset, client_indices[i]),
        batch_size=batch_size,
        shuffle=True
    ) for i in range(num_clients)
]
test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize global model (SNNNet)
global_model = SNNNet().to(device)

# Federated Learning training loop
for round_idx in range(num_rounds):
    print(f"\n--- Round {round_idx + 1} ---")
    client_models = []
    client_losses = []
    client_accuracies = []

    # Local training for each client
    for client_id in range(num_clients):
        local_model = SNNNet().to(device)
        local_model.load_state_dict(global_model.state_dict())  # Initialize with global model

        # Train the local model
        local_state_dict, loss, acc = train_client(
            local_model, train_loaders[client_id], local_epochs, lr, T
        )

        client_models.append(local_state_dict)
        client_losses.append(loss)
        client_accuracies.append(acc)

        print(f"Client {client_id + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

        # Log client results to the file
        with open(results_file, "a") as f:
            f.write(f"{round_idx + 1}, {client_id + 1}, {loss:.4f}, {acc:.4f}\n")

    # Aggregate client models to update the global
    aggregate_models(global_model, client_models)

    # evaluat
    global_loss, global_acc = test_model(global_model, test_loader, T)
    print(f"Global Model: Loss = {global_loss:.4f}, Accuracy = {global_acc:.4f}")


    with open(results_file, "a") as f:
        f.write(f"{round_idx + 1}, Global, {global_loss:.4f}, {global_acc:.4f}\n")

# save final model
final_model_path = "30epoch_final_global_model_snn_noniid_5.pth"
torch.save(global_model.state_dict(), final_model_path)
print(f"Final global model saved as '{final_model_path}'.")
