'''
This code file is used for federated learning,
where the central model is trained using average aggregation.
You can train with either IID or Non-IID data distributions.
By default, the code uses the Non-IID setting.
The training dataset is the SVHN dataset.

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
from models.CNN import CNN
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

def load_svhn():
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])
    trainset = datasets.SVHN(root="../data/data_SVHN", split='train', download=False, transform=transform)
    testset = datasets.SVHN(root="../data/data_SVHN", split='test', download=False, transform=transform)
    return trainset, testset



# This function is used in federated learning experiments for
# simulating IID and Non-IID client datasets.
def split_data(dataset, num_clients, iid, alpha=1.0, seed=42):
    """
    When alpha → 0, a client may contain only a very small number of classes.
    When alpha → +∞, the distribution tends to be uniform (similar to IID).
      - alpha = 0.1: very large differences among clients
                     (e.g., some clients may only contain 1–2 classes).
      - alpha = 1.0: moderate heterogeneity (default).
      - alpha = 10.0: close to IID distribution.

    Split Non-IID data based on Dirichlet distribution by class proportion.

    Args:
        dataset: dataset containing data and labels
                 (assumed accessible via dataset.labels or dataset.targets)
        num_clients: number of clients
        alpha: parameter of the Dirichlet distribution that controls
               the degree of data heterogeneity across clients
               (smaller alpha → higher heterogeneity)
        seed: random seed for reproducibility

    Returns:
        A list of sample indices for each client
    """

    indices = list(range(len(dataset)))
    if iid:
        random.shuffle(indices)
        return [indices[i::num_clients] for i in range(num_clients)]

    else:
        np.random.seed(seed)
        random.seed(seed)

        # Get labels and convert to numpy array
        if hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        elif hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            raise ValueError("Dataset must have 'labels' or 'targets' attribute")

        n_classes = len(np.unique(labels))
        class_indices = [np.where(labels == c)[0] for c in range(n_classes)]  # group indices by class

        # Generate client allocation proportions for each class (Dirichlet sampling)
        distribution = np.random.dirichlet([alpha] * num_clients, n_classes)

        client_indices = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            class_idcs = class_indices[c]
            np.random.shuffle(class_idcs)
            proportions = distribution[c]  # proportion of current class for each client
            split_points = (np.cumsum(proportions) * len(class_idcs)).astype(int)[:-1]
            client_splits = np.split(class_idcs, split_points)

            for client_id in range(num_clients):
                client_indices[client_id].extend(client_splits[client_id].tolist())

        # Shuffle the sample order within each client
        for client_id in range(num_clients):
            np.random.shuffle(client_indices[client_id])

        return client_indices


# 训练一个本地客户端
def train_client(model, train_loader, epochs, lr):
    """
    Train a local client model using its own dataset.

    Args:
        model: PyTorch model to be trained locally.
        train_loader: DataLoader containing the client’s local training data.
        epochs: Number of local training epochs per federated round.
        lr: Learning rate for the optimizer.

    Returns:
        model.state_dict(): The updated model parameters after local training.
        avg_loss (float): Average training loss across the client’s dataset.
        accuracy (float): Training accuracy on the client’s dataset.
    """
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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


def aggregate_models(global_model, client_models):
    """
    Aggregate local client model parameters to update the global model
    using Federated Averaging (FedAvg).

    Args:
        global_model: The global model to be updated.
        client_models: A list of state_dicts containing parameters from each client.

    Returns:
        None. The global_model is updated in place with averaged parameters.
    """
    global_state_dict = OrderedDict()
    for key in global_model.state_dict().keys():
        global_state_dict[key] = torch.mean(
            torch.stack([client_models[i][key] for i in range(num_clients)]), dim=0
        )
    global_model.load_state_dict(global_state_dict)


def test_model(model, test_loader):
    """
    Evaluate the performance of a model on the test dataset.

    Args:
        model: The PyTorch model to be evaluated.
        test_loader: DataLoader containing the test dataset.

    Returns:
        avg_loss (float): Average test loss across the dataset.
        accuracy (float): Classification accuracy on the test dataset.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


'''
Step 1: Set different parameters in the code, 
such as whether to use IID data distribution, 
the number of federated training rounds, 
the number of local epochs per round, 
and the learning rate.
'''
# setting of Federated Learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 5  # number of client model
iid = False  # data distribution
num_rounds = 30  # epochs of federated framework
local_epochs = 2  # epochs of local train
batch_size = 64
lr = 0.01  # Learning rate


'''
Step 2: load training dataset: SVHN
'''
trainset, testset = load_svhn()


'''
Step 3: split the data into IID or Non-IID.
'''
client_indices = split_data(trainset, num_clients, iid,alpha=1.0, seed=42)


'''
Step 4: Train model and save it. Federated Learning Training Loop
'''

# Define result file and remove it if exists
results_file = "30epoch_federated_results_noniid_5.txt"
if os.path.exists(results_file):
    os.remove(results_file)

# Initialize result file with header
with open(results_file, "w") as f:
    f.write("Round, Client, Loss, Accuracy\n")

# Prepare DataLoaders for clients and test set
train_loaders = [
    data.DataLoader(
        data.Subset(trainset, client_indices[i]),
        batch_size=batch_size,
        shuffle=True
    ) for i in range(num_clients)
]
test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize global model
global_model = CNN().to(device)

# Start federated training
for round_idx in range(num_rounds):
    print(f"\n--- Round {round_idx + 1} ---")

    client_models = []
    client_losses = []
    client_accuracies = []

    # Local training for each client
    for client_id in range(num_clients):
        local_model = CNN().to(device)
        local_model.load_state_dict(global_model.state_dict())

        local_state_dict, loss, acc = train_client(
            local_model, train_loaders[client_id], local_epochs, lr
        )

        client_models.append(local_state_dict)
        client_losses.append(loss)
        client_accuracies.append(acc)

        print(f"Client {client_id + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

        # Log client results
        with open(results_file, "a") as f:
            f.write(f"{round_idx + 1}, {client_id + 1}, {loss:.4f}, {acc:.4f}\n")

    # Aggregate local models into the global model
    aggregate_models(global_model, client_models)

    # Evaluate global model
    global_loss, global_acc = test_model(global_model, test_loader)
    print(f"Global Model: Loss = {global_loss:.4f}, Accuracy = {global_acc:.4f}")

    # Log global results
    with open(results_file, "a") as f:
        f.write(f"{round_idx + 1}, Global, {global_loss:.4f}, {global_acc:.4f}\n")

# Save the final global model
final_model_path = "30epoch_final_global_model_noniid_5.pth"
torch.save(global_model.state_dict(), final_model_path)
print(f"Final global model saved as '{final_model_path}'.")
