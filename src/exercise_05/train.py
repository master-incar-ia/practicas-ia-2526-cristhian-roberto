from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Importamos nuestras clases personalizadas
from .dataset import CIFAR10Dataset
from .model import FullyConnectedClassifier

num_epochs = 200  # Clasificación requiere menos épocas con buen LR que la regresión simple


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(output_folder: Path, device: torch.device):
    # 1. Definir transformaciones (Data Augmentation y Normalización)
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # 2. Cargar Dataset y Split (Entrenamiento y Validación)
    full_dataset = CIFAR10Dataset(
        root="./data", train=True, transform=transform_train, download=True
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Aplicar transformación de validación (sin augmentation) al set de val
    val_dataset.dataset.transform = transform_val

    # 3. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # 4. Instanciar Modelo, Pérdida y Optimizador
    model = FullyConnectedClassifier(input_dim=3 * 32 * 32, hidden_dim=1024, output_dim=10).to(
        device
    )
    criterion = nn.CrossEntropyLoss()  # Estándar para clasificación multiclase
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 5. NUEVO: Programador de Tasa de Aprendizaje (Scheduler)
    # Esto reduce el LR cuando el entrenamiento se estanca para "afinar" la puntería
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # 6. Bucle de entrenamiento
    # num_epochs = num_epochs
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 7. Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Calcular precisión (Accuracy)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        acc = 100.0 * correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(acc)

        print(f"Loss: T {avg_train_loss:.4f} | V {avg_val_loss:.4f} - Acc: {acc:.2f}%")

        scheduler.step(acc)  # <--- El scheduler actúa basándose en la precisión (acc) obtenida

        # Guardar mejor modelo basado en Accuracy
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), output_folder / f"best_model_fc_{num_epochs}_epochs.pth")

    # 8. Gráficas finales
    plot_results(history, output_folder, num_epochs)
    return num_epochs


def plot_results(history, output_folder, num_epochs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Acc", color="green")
    plt.legend()
    plt.title("Accuracy (%)")

    plt.savefig(output_folder / f"metrics_plot_{num_epochs}_epochs.png")
    plt.show()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    device = get_device()
    torch.manual_seed(42)
    train_model(output_folder, device)
