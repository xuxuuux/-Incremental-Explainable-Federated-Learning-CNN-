'''
This code file provides LIME-based visual explanations
for the federated CNN model after incremental training.

In this code, you can freely choose to use digits 0–9 from either
the MNIST or SVHN dataset, view the predictions of the
federated CNN model, and generate LIME-based visual explanations.
All of these interactions are carried out through a human-computer dialogue.

I have saved an incremental trained federated CNN model in the
**save\_models\_pth** folder as **final\_updated\_FL\_model.pth**.
'''

import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
from models.CNN import CNN


'''
Step 1: Settings.
load the final model after incremental FL learning.
'''
MODEL_PATH = "../save_models_pth/final_updated_FL_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Step 2: Select the Image from the dataset you want to predict and 
perform LIME visualization through human-computer interaction and dialogue.
'''
dataset_choice = input("Please choose a dataset (MNIST / SVHN): ").strip().upper()

if dataset_choice == "MNIST":
    # User specifies which digit to inspect
    try:
        target_label = int(input("Please enter the digit to predict and inspect (0-9): ").strip())
        if not (0 <= target_label <= 9):
            raise ValueError
    except ValueError:
        print("Please enter a valid digit (0-9)")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),  # Convert MNIST to 3 channels
        transforms.ToTensor(),
    ])
    test_data = datasets.MNIST(
        root="../data/data_MNIST",
        train=False,
        download=False,
        transform=transform
    )

elif dataset_choice == "SVHN":
    target_label = 3  # Default digit for SVHN
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    test_data = datasets.SVHN(
        root="../data/data_SVHN",
        split="test",
        download=False,
        transform=transform
    )

else:
    print("Invalid input. Please enter MNIST or SVHN.")
    sys.exit(1)

'''
Step 3: Prepare Target Subset
'''
indices_target = [i for i, (_, label) in enumerate(test_data) if int(label) == target_label]
subset_target = Subset(test_data, indices_target)
loader_target = DataLoader(subset_target, batch_size=len(subset_target), shuffle=False)


# Load Trained Model
model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Get all target samples
images, labels = next(iter(loader_target))

'''
Step 4: LIME Prediction Function
'''
def batch_predict(images):
    """
    Prediction function for LIME. Converts input images into tensors,
    feeds them into the model, and returns probabilities.
    """
    images = torch.tensor(images.transpose((0, 3, 1, 2)), dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(images)
    return outputs.cpu().numpy()

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()


current_idx = 0
total_samples = len(images)

while current_idx < total_samples:
    image_tensor = images[current_idx]
    label_true = int(labels[current_idx])
    image_np = image_tensor.numpy().transpose(1, 2, 0)

    # Model prediction
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        label_pred = output.argmax(dim=1).item()
        probs = torch.softmax(output, dim=1)[0]

    # Show original image and prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(image_np, cmap="gray")
    plt.title(f"True label: {label_true}, Prediction: {label_pred}", fontsize=12)
    plt.axis("off")
    plt.show()

    print(f"\n=== Sample {current_idx + 1}/{total_samples} ===")
    print(f"True Label: {label_true}")
    print(f"Predicted Label: {label_pred}")
    print(f"Prediction Probability: {probs[label_pred]:.4f}")
    print("All Class Probabilities:", {i: f"{probs[i]:.4f}" for i in range(10)})

    # LIME explanation
    explanation = explainer.explain_instance(
        image_np,
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=10000,
        segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10)
    )

    try:
        temp, mask = explanation.get_image_and_mask(
            label_pred,
            positive_only=True,
            num_features=3,
            hide_rest=False
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(mark_boundaries(temp, mask))
        plt.title("Important Features for Model Decision", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    except KeyError:
        print(f"Predicted label {label_pred} not in LIME's top_labels")
        print(f"LIME generated top_labels: {explanation.top_labels}")

    # Ask user if continue
    user_input = input("Do you want to continue to the next sample? (y/n): ").lower()
    if user_input != "y":
        break

    current_idx += 1

print(f"\nInspected {current_idx + 1} samples out of {total_samples} samples labeled {target_label}.")
