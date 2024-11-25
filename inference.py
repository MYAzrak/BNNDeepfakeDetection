import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from PIL import Image

from BNext_model import BNext
from config import get_path


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def load_model(model_path, device):
    model = nn.DataParallel(BNext(num_classes=2)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def plot_predictions(model, image_paths, transform, device):
    # Load dataset to get class mapping
    dataset = datasets.ImageFolder(root=get_path("test_dir"))
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    classes = [idx_to_class[0], idx_to_class[1]]

    n_images = len(image_paths)
    fig, axs = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    fig.suptitle("Image Predictions", fontsize=16)

    for i, image_path in enumerate(image_paths):
        # Determine actual label from file path
        actual_label = os.path.basename(os.path.dirname(image_path))

        # Preprocess image
        image_tensor = preprocess_image(image_path, transform).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = classes[predicted.item()]

        # Load and display image
        image = Image.open(image_path)

        # Plot image
        if n_images == 1:
            current_ax = axs
        else:
            current_ax = axs[i]
        current_ax.imshow(image)
        current_ax.axis("off")

        # Add prediction text
        title = f"Actual: {actual_label}\nPredicted: {predicted_label}"
        current_ax.set_title(title)

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = get_path("model_path")
    model = nn.DataParallel(BNext(num_classes=2)).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dir = get_path("test_dir")

    image_paths = [
        os.path.join(test_dir, "REAL", r"0000 (2).jpg"),
        os.path.join(test_dir, "REAL", r"0000 (3).jpg"),
        os.path.join(test_dir, "REAL", r"0000 (4).jpg"),
        os.path.join(test_dir, "FAKE", r"0 (2).jpg"),
        os.path.join(test_dir, "FAKE", r"0 (3).jpg"),
        os.path.join(test_dir, "FAKE", r"0 (4).jpg"),
    ]

    plot_predictions(model, image_paths, transform, device)


if __name__ == "__main__":
    main()
