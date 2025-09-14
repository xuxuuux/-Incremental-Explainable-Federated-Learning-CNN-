'''
This code file is used to train the baseline model and visualize
the performance of the baseline CNN model on the SVHN dataset
using a confusion matrix.

In this project, I have already trained the baseline code file,
and the model is saved as **baseline\_svhn\_cnn.pth**
in the **save\_models\_pth** folder. If you want to retrain
the baseline model, simply uncomment the training code.

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models.CNN import CNN
import torchvision
from lime import lime_image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Step 1: load training data.
'''
# Data preprocessing:
# 1. Resize all images to 32x32 pixels to match the CNN input size.
# 2. Convert images to PyTorch tensors.
transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])

# load dataset(SVHN)
train_dataset = datasets.SVHN(root='../data/data_SVHN', split='train', download=False, transform=transform)
test_dataset = datasets.SVHN(root='../data/data_SVHN', split='test', download=False, transform=transform)

# Create dataloader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



'''
Step 2: load baseline CNN Model and train it.
'''
model = CNN().to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Record the effect of each round of training
num_epochs = 30
train_losses = []
train_accuracies = []
test_accuracies = []


'''

# #train the simple CNN and save it.
#
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # 统计信息
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     # 计算训练指标
#     train_loss = running_loss / len(train_loader)
#     train_acc = 100 * correct / total
#     train_losses.append(train_loss)
#     train_accuracies.append(train_acc)
#
#     # 测试阶段
#     model.eval()
#     test_correct = 0
#     test_total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             test_total += labels.size(0)
#             test_correct += (predicted == labels).sum().item()
#
#     test_acc = 100 * test_correct / test_total
#     test_accuracies.append(test_acc)
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], "
#           f"Train Loss: {train_loss:.4f}, "
#           f"Train Acc: {train_acc:.2f}%, "
#           f"Test Acc: {test_acc:.2f}%")
#
# # 保存模型
# torch.save(model.state_dict(), "svhn_cnn_30epochs.pth")
#
#
# # 可视化训练过程
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Training Loss')
# plt.title("Training Loss")
# plt.xlabel("Epoch")
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Training Accuracy')
# plt.plot(test_accuracies, label='Test Accuracy')
# plt.title("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()
'''


'''
Step 3: evaluate the baseline model by Confusion Matrix & F1 score.
'''
model = CNN().to(device)
model.load_state_dict(torch.load("..\save_models_pth\\baseline_svhn_cnn.pth", map_location=device))
model.eval()

# Calculate the confusion matrix
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


conf_matrix = confusion_matrix(y_true, y_pred)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # 归一化

# Calculate the F1 score
f1_scores = f1_score(y_true, y_pred, average=None)

# Visualized confusion matrix
plt.figure(figsize=(10, 8))

sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label',fontsize=16)
plt.ylabel('True Label',fontsize=16)
plt.title('Normalized Confusion Matrix(simple CNN; Baseline)',fontsize=16)
plt.show()

