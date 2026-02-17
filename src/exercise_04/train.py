from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from .dataset import CIFAR10Dataset
from .model import CNNClassifier


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def train_model(output_folder: Path, device: torch.device):
    # Transform: ToTensor es OBLIGATORIO (si no, DataLoader recibe PIL y peta)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    # Dataset (train split de CIFAR10)
    dataset = CIFAR10Dataset(root="./data", train=True, transform=transform, download=True)

    # Split train/val/test desde train (si quieres test oficial: usa train=False aparte)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    g = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=g
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=pin_memory)

    # Modelo
    model = CNNClassifier(output_dim=10).to(device)

    # Loss y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    num_epochs = 60
    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model_cnn.pth"

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation loss
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(inputs)
                loss = criterion(logits, targets)
                vloss += loss.item()

        val_loss = vloss / len(val_loader)
        val_losses.append(val_loss)

        # Validation accuracy
        val_acc = accuracy(model, val_loader, device)
        val_accs.append(val_acc)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f}"
            )

    print(f"Best val loss: {best_val_loss:.4f} -> saved to {best_model_path}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(output_folder / "loss_plot.png", dpi=200)
    plt.close()

    # Plot val accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), val_accs, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(output_folder / "val_acc_plot.png", dpi=200)
    plt.close()

    return best_model_path


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    train_model(output_folder, device=device)
