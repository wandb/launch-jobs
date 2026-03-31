"""Example training job"""
# MNIST code heavily inspired by https://www.geeksforgeeks.org/fashion-mnist-with-python-keras-and-deep-learning/
import argparse
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import wandb


LABELS = [
    "t_shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boots",
]


def model_arch() -> nn.Module:
    """Define the architecture of the model"""
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )


def train(project: Optional[str], entity: Optional[str], **kwargs: Any):
    run = wandb.init(project=project, entity=entity, config={
        "epochs": 10,
        "learning_rate": 0.001,
        "steps_per_epoch": 10,
    })

    train_config = run.config
    epochs = train_config.get("epochs", 10)
    learning_rate = train_config.get("learning_rate", 0.001)
    steps_per_epoch = train_config.get("steps_per_epoch", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root="/tmp/data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="/tmp/data", train=False, download=True, transform=transform)

    # Cut down for speed (match original 1200 samples)
    train_dataset = Subset(train_dataset, range(1200))

    # log some images
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        img, _ = train_dataset[i]
        plt.imshow(img.squeeze(), cmap=plt.get_cmap("gray"))
    wandb.log({"chart": plt})
    plt.clf()

    train_loader = DataLoader(train_dataset, batch_size=max(1, 1200 // steps_per_epoch), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=9, shuffle=False)

    model = model_arch().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for step, (X, y) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

        acc = correct / total if total > 0 else 0
        wandb.log({"epoch": epoch + 1, "loss": total_loss / steps_per_epoch, "sparse_categorical_accuracy": acc})

    # Predictions on first 9 test samples
    model.eval()
    test_images, _ = next(iter(test_loader))
    with torch.no_grad():
        preds = model(test_images.to(device)).argmax(1).cpu()
    pred_labels = [LABELS[p] for p in preds]

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[i].squeeze(), cmap=plt.get_cmap("gray"))
        plt.title(f"Pred: {pred_labels[i]}")
    wandb.log({"prediction-chart": plt})

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", "-e", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default=None)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
