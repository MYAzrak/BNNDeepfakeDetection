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

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Preprocessing (Normalization) and Augmentation
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = r"C:\Users\monsi\Downloads\temp\CIFAKE-Real-and-AI-Generated-Synthetic-Images\DATASET\train"
    test_dir = r"C:\Users\monsi\Downloads\temp\CIFAKE-Real-and-AI-Generated-Synthetic-Images\DATASET\test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Taking a small subset for a faster process
    num_samples = 10
    random_indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    train_subset = Subset(train_dataset, random_indices)

    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    test_subset = Subset(test_dataset, random_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)

    num_epochs = 1

    train_history = []
    val_history = []

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criteria(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        # Validation
        with torch.no_grad():
            val_loss = 0
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criteria(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch [{epoch}], train loss: {train_loss/len(train_dataset)}, val loss: {val_loss/len(test_dataset)}"
        )
        train_history += [train_loss / len(train_dataset)]
        val_history += [val_loss / len(test_dataset)]
    print("Finished Training")

    torch.save(model.state_dict(), "bnext_trained.pth")


if __name__ == "__main__":
    main()
