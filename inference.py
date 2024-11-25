import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import os

from BNext_model import BNext


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def load_model(model_path, device):
    model = nn.DataParallel(BNext(num_classes=2)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def inference_on_examples(model, image_paths, transform, device):
    dataset = datasets.ImageFolder(
        root=r"C:\Users\monsi\Downloads\temp\CIFAKE-Real-and-AI-Generated-Synthetic-Images\DATASET\test"
    )
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    classes = [
        idx_to_class[0],
        idx_to_class[1],
    ]
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path, transform).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            print(
                f"Image: {os.path.basename(image_path)} - Predicted: {classes[predicted.item()]}"
            )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "best_model.pth"
    model = nn.DataParallel(BNext(num_classes=2)).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    # Define preprocessing transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dir = r"C:\Users\monsi\Downloads\temp\CIFAKE-Real-and-AI-Generated-Synthetic-Images\DATASET\test"

    image_paths = [
        os.path.join(test_dir, "REAL", r"0000 (2).jpg"),
        os.path.join(test_dir, "REAL", r"0000 (3).jpg"),
        os.path.join(test_dir, "REAL", r"0000 (4).jpg"),
        os.path.join(test_dir, "FAKE", r"0 (2).jpg"),
        os.path.join(test_dir, "FAKE", r"0 (3).jpg"),
        os.path.join(test_dir, "FAKE", r"0 (4).jpg"),
    ]

    inference_on_examples(model, image_paths, transform, device)


if __name__ == "__main__":
    main()
