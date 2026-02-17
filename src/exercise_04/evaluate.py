from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

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
def evaluate_split(model, loader, device, criterion=None, num_classes=10):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        preds = logits.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item()

        for t, p in zip(targets.view(-1), preds.view(-1)):
            conf[t.long(), p.long()] += 1

    acc = correct / total if total > 0 else 0.0
    avg_loss = (total_loss / len(loader)) if (criterion is not None and len(loader) > 0) else None
    return acc, avg_loss, conf


def save_confusion_matrix(
    conf, class_names, png_path: Path, csv_path: Path | None = None, title="Confusion Matrix"
):
    conf_np = conf.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_np,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    if csv_path is not None:
        pd.DataFrame(conf_np, index=class_names, columns=class_names).to_csv(csv_path)


def save_metrics_table_png(metrics_df: pd.DataFrame, png_path: Path, title: str = "Metrics"):
    """Guarda un PNG con una tabla a partir de un DataFrame (p.ej. metrics.csv)."""
    df = metrics_df.copy()

    if "accuracy" in df.columns:
        df["accuracy"] = df["accuracy"].map(lambda x: f"{x:.4f}")
    if "loss" in df.columns:
        df["loss"] = df["loss"].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(6, 2 + 0.4 * len(df)))
    ax.axis("off")
    ax.set_title(title, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)


def save_split_metrics_png(split_row: dict, png_path: Path, title: str):
    """
    Guarda un PNG por split (train/validation/test) con accuracy y loss.
    Lo dibujo como mini-barchart (2 barras) para que sea fácil de leer.
    """
    labels = ["accuracy", "loss"]
    values = [float(split_row["accuracy"]), float(split_row["loss"])]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, values)  
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)

    
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close(fig)


def main():
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(42)
    device = get_device("auto")
    print(f"Using device: {device}")

  
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

   
    dataset = CIFAR10Dataset(root="./data", train=True, transform=transform, download=True)

 
    g = torch.Generator().manual_seed(42)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=g
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=False, pin_memory=pin_memory, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, pin_memory=pin_memory, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, pin_memory=pin_memory, num_workers=2
    )


    model = CNNClassifier(output_dim=10).to(device)

    best_model_path = output_folder / "best_model_cnn.pth"  
    if not best_model_path.exists():
        raise FileNotFoundError(f"No existe el archivo de pesos: {best_model_path}")

    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    class_names = dataset.data.classes  

    metrics_rows = []

    for split_name, loader in [
        ("train", train_loader),
        ("validation", val_loader),
        ("test", test_loader),
    ]:
        acc, loss, conf = evaluate_split(model, loader, device, criterion=criterion, num_classes=10)
        print(f"{split_name}: acc={acc:.4f} loss={loss:.4f}")

      
        save_confusion_matrix(
            conf,
            class_names,
            png_path=output_folder / f"confusion_{split_name}.png",
            csv_path=output_folder / f"confusion_{split_name}.csv",
            title=f"Confusion Matrix ({split_name})",
        )

        metrics_rows.append({"split": split_name, "accuracy": acc, "loss": loss})

    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_folder / "metrics.csv", index=False)

    
    save_metrics_table_png(
        metrics_df,
        png_path=output_folder / "metrics.png",
        title="Metrics (train / validation / test)",
    )

    
    for row in metrics_rows:
        split = row["split"]
        save_split_metrics_png(
            row,
            png_path=output_folder / f"metrics_{split}.png",
            title=f"Metrics ({split})",
        )

    print(f"Evaluación completada. Resultados en: {output_folder.resolve()}")
    print("Guardado: metrics.csv + metrics.png + metrics_{train,validation,test}.png")
    print("Guardado: confusion_{train,validation,test}.png/.csv")


if __name__ == "__main__":
    main()
