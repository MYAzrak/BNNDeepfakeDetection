import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from BNext_model import BNext


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(BNext(num_classes=2)).to(device)

    model_path = "best_model.pth"
    model.load_state_dict(torch.load(model_path, weights_only=False))

    # Preprocessing (Normalization)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load test dataset
    test_dir = r"C:\Users\monsi\Downloads\temp\CIFAKE-Real-and-AI-Generated-Synthetic-Images\DATASET\test"
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Taking a small subset for a faster process
    num_samples = 10
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    test_subset = Subset(test_dataset, random_indices)

    test_loader = DataLoader(test_subset, batch_size=512, shuffle=False, num_workers=4)

    # Set model to evaluation mode
    model.eval()

    # Initialize variables to track correct predictions and total number of samples
    correct = 0
    total = 0

    # Lists to store true labels and predictions for precision, recall, and F1-score
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    # Print results
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")


if __name__ == "__main__":
    main()
