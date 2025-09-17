import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from spikingjelly.activation_based import neuron, functional, encoding
from models.SNN import SNNNet  # 使用SNN模型

'''
Step 1: Load Pretrained Global Model
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNNNet()
model.load_state_dict(torch.load("final_global_model.pth", map_location=device))
model.to(device)


'''
Step 2: Split MNIST Dataset (IID / non-IID)
'''
def split_mnist_data(num_clients=5, iid=True):
    """
    Split MNIST dataset into IID or non-IID subsets for clients.
    Data is resized to 32x32 and converted to 3 channels.

    Args:
        num_clients (int): Number of federated clients.
        iid (bool): Whether to split data in an IID fashion.

    Returns:
        List of Subset datasets for each client.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),               # Resize to 32x32
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
    ])

    mnist_dataset = datasets.MNIST(
        root="../data/data_MNIST",
        train=True,
        download=False,
        transform=transform
    )
    data_per_client = len(mnist_dataset) // num_clients

    if iid:
        indices = list(range(len(mnist_dataset)))
        random.shuffle(indices)
        client_datasets = [
            data.Subset(mnist_dataset, indices[i * data_per_client:(i + 1) * data_per_client])
            for i in range(num_clients)
        ]
    else:
        labels = np.array(mnist_dataset.targets)
        sorted_indices = np.argsort(labels)
        client_datasets = [
            data.Subset(mnist_dataset, sorted_indices[i * data_per_client:(i + 1) * data_per_client])
            for i in range(num_clients)
        ]

    return client_datasets


'''
Step 3: Knowledge Distillation Loss
'''
def knowledge_distillation_loss(new_outputs, old_outputs, temperature=2.0):
    """
    Compute knowledge distillation loss (KL divergence between softened outputs).

    Args:
        new_outputs: Predictions from the new model.
        old_outputs: Predictions from the previous global model.
        temperature (float): Temperature parameter for softening.

    Returns:
        torch.Tensor: Distillation loss value.
    """
    soft_targets = nn.functional.softmax(old_outputs / temperature, dim=1)
    soft_outputs = nn.functional.log_softmax(new_outputs / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(soft_outputs, soft_targets) * (temperature ** 2)

'''
Step 4: Model Evaluation
'''
def evaluate(model, test_loader, T=50):
    """
    Evaluate model performance on a given test dataset.

    Args:
        model: PyTorch model to evaluate.
        test_loader: DataLoader for test dataset.
        T: Number of simulation time steps for SNN.

    Returns:
        acc (float): Accuracy.
        avg_loss (float): Average cross-entropy loss.
    """
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    encoder = encoding.PoissonEncoder()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Convert static image to [T, batch, C, H, W] spike sequence
            x_seq = torch.stack([encoder(inputs) for _ in range(T)])

            outputs = model(x_seq)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(test_loader)
    return acc, avg_loss


'''
Step 5: Federated Update (with knowledge distillation)
'''
def federated_update(model, client_datasets, num_rounds=6, epochs=1, lr=0.01, lambda_kd=0.5, T=50):
    """
    Perform federated training with knowledge distillation (no replay from SVHN).

    Args:
        model: Pre-trained global model (SNN).
        client_datasets: List of client datasets (MNIST).
        num_rounds (int): Number of federated rounds.
        epochs (int): Local training epochs.
        lr (float): Learning rate.
        lambda_kd (float): Weight for KD loss.
        T (int): Number of simulation time steps for SNN.

    Returns:
        global_model: Updated global model after federated training.
    """
    global_model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()

    # Load MNIST test set
    mnist_test = datasets.MNIST(
        root="../data/data_MNIST",
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    )
    mnist_test_loader = data.DataLoader(mnist_test, batch_size=32, shuffle=False)

    # Federated rounds
    for round_idx in range(num_rounds):
        local_models = []

        for dataset in client_datasets:
            local_model = copy.deepcopy(global_model)
            local_model.train()
            optimizer = optim.SGD(local_model.parameters(), lr=lr)
            dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

            for _ in range(epochs):
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # Convert static image to [T, batch, C, H, W] spike sequence
                    x_seq = torch.stack([encoding.PoissonEncoder()(inputs) for _ in range(T)])

                    outputs = local_model(x_seq)
                    loss = criterion(outputs, labels)

                    # Knowledge distillation loss
                    with torch.no_grad():
                        old_outputs = global_model(x_seq)
                    kd_loss = knowledge_distillation_loss(outputs, old_outputs)

                    # Combined loss
                    total_loss = loss + lambda_kd * kd_loss
                    total_loss.backward()
                    optimizer.step()

            local_models.append(local_model.state_dict())

        # Federated averaging
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([local_model[key] for local_model in local_models], dim=0).mean(dim=0)
        global_model.load_state_dict(global_dict)

        # Evaluate on MNIST
        mnist_acc, mnist_loss = evaluate(global_model, mnist_test_loader, T)
        print(
            f"Round {round_idx + 1}/{num_rounds} - "
            f"MNIST Acc: {mnist_acc:.4f}, Loss: {mnist_loss:.4f}"
        )

    return global_model


'''
Run Federated Incremental Training with knowledge distillation
'''

if __name__ == "__main__":
    # Split MNIST for clients (set iid=False for non-IID)
    client_datasets = split_mnist_data(iid=True)

    # Perform federated update with KD
    updated_model = federated_update(model, client_datasets)

    # Save the updated global model
    save_path = "updated_model.pth"
    torch.save(updated_model.state_dict(), save_path)
    print(f"Updated global model saved at: {save_path}")
