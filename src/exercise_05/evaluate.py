from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

# Importamos las clases locales
from .dataset import CIFAR10Dataset
from .model import FullyConnectedClassifier
from .train import num_epochs


def evaluate_classification(loader, model, device, output_folder, class_names):
    model.eval()
    all_preds = []
    all_targets = []

    # 1. Fase de Inferencia
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Obtenemos el índice de la probabilidad más alta (la clase predicha)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 2. Cálculo de métricas de clasificación
    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()

    print(f"\n--- Métricas Globales ---")
    print(f"Precisión Total (Accuracy): {acc:.4f}")
    print(df_report)

    # 3. Generación de la Matriz de Confusión
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.title("Matriz de Confusión - CIFAR10")
    plt.savefig(output_folder / f"confusion_matrix_{num_epochs}_epochs.png")
    plt.show()

    return df_report


def save_metrics_table(df_report, filepath):
    # Seleccionamos solo las métricas principales para el reporte visual
    metrics_to_show = df_report.iloc[:-3, :3]  # Precision, Recall, F1 por clase

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")
    ax.table(
        cellText=metrics_to_show.values.round(3),
        colLabels=metrics_to_show.columns,
        rowLabels=metrics_to_show.index,
        cellLoc="center",
        loc="center",
    )
    plt.title("Resumen de Clasificación por Categoría", pad=20)
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformación idéntica a la de validación/test en train.py
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Cargamos el dataset de TEST original
    test_dataset = CIFAR10Dataset(root="./data", train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    class_names = test_dataset.data.classes

    # Cargamos el modelo entrenado
    model = FullyConnectedClassifier(input_dim=3 * 32 * 32, hidden_dim=1024, output_dim=10).to(
        device
    )
    model_path = output_folder / f"best_model_fc_{num_epochs}_epochs.pth"

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Pesos del modelo cargados correctamente.")

        # Ejecutamos evaluación
        df_metrics = evaluate_classification(test_loader, model, device, output_folder, class_names)

        # Guardamos resultados
        df_metrics.to_csv(output_folder / f"classification_metrics_{num_epochs}_epochs.csv")
        save_metrics_table(
            df_metrics, output_folder / f"final_metrics_table_{num_epochs}_epochs.png"
        )

        print(f"\nEvaluación finalizada. Resultados guardados en {output_folder}")
    else:
        print(
            f"❌ Error: No se encontró el archivo de pesos en {model_path}. Ejecuta train.py primero."
        )
