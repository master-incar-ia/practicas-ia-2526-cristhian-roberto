from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, random_split

# Importamos las clases definidas anteriormente
from .dataset import QuadraticNoiseDataset
from .model import NonlinearRegressor


def evaluate_and_plot(loader, model, dataset_name, output_folder):
    model.eval()
    all_inputs = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            all_inputs.append(inputs.numpy())
            all_outputs.append(outputs.numpy())
            all_targets.append(targets.numpy())

    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # DataFrame para análisis
    df = pd.DataFrame(
        data=np.array(
            [all_inputs.flatten(), all_targets.flatten(), all_outputs.flatten()]
        ).transpose(),
        columns=["x", "y_true", "y_pred"],
    )

    # --- CÁLCULO DE MÉTRICAS ---
    r2 = 1 - np.sum((all_targets - all_outputs) ** 2) / np.sum(
        (all_targets - np.mean(all_targets)) ** 2
    )
    mae = np.mean(np.abs(all_targets - all_outputs))
    mse = np.mean((all_targets - all_outputs) ** 2)
    # RMSE: Más interpretable (en la escala de 'y')
    rmse = np.sqrt(mse)
    # Explained Variance Score: Mide qué tanto captura la "forma" de la parábola
    # ignorando errores de desplazamiento sistemático.
    var_error = np.var(all_targets - all_outputs)
    var_true = np.var(all_targets)
    explained_variance = 1 - (var_error / var_true)

    metrics = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE (Mejor)": rmse,
        "Expl_Variance": explained_variance,
    }

    print(f"Evaluation metrics for {dataset_name} dataset:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # --- GRÁFICOS ---
    # 1. Regresión y_true vs y_pred
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x="y_true", y="y_pred", scatter_kws={"alpha": 0.3})
    plt.title(f"Regression Check: {dataset_name}")
    plt.savefig(output_folder / f"{dataset_name}_regression_plot.png")
    plt.show()
    plt.close()

    # 2. Ajuste de la curva (X vs Y) - Crucial para la función cuadrática
    plt.figure(figsize=(10, 6))
    # Ordenamos para que la línea roja sea continua
    df_sorted = df.sort_values(by="x")
    sns.scatterplot(data=df, x="x", y="y_true", label="Real (con ruido)", alpha=0.4)
    plt.plot(
        df_sorted["x"], df_sorted["y_pred"], color="red", label="Predicción (Curva)", linewidth=2.5
    )
    plt.title(f"Curva de Ajuste Cuadrático - {dataset_name}")
    plt.legend()
    plt.savefig(output_folder / f"{dataset_name}_curve_fit.png")
    plt.show()
    plt.close()

    return metrics


def save_metrics_as_picture(metrics, filepath):
    # Create a DataFrame
    df = pd.DataFrame(metrics)

    # Round the values to 3 decimal places
    df = df.round(3)

    # Plot the table and save as an image
    fig, ax = plt.subplots(figsize=(8, 2))  # set size frame
    ax.axis("tight")
    ax.axis("off")
    ax.table(
        cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center"
    )

    # Save the plot as an image
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    # Set the seed for reproducibility
    torch.manual_seed(42)
    # Create an instance of the dataset
    dataset = QuadraticNoiseDataset(size=2000)

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for the datasets
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Load the best model weights
    model = NonlinearRegressor(input_dim=1, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(output_folder / "best_model_2.pth"))

    metrics = {}
    # Evaluate and plot for train, validation and test datasets
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder)
    metrics["validation"] = evaluate_and_plot(val_loader, model, "validation", output_folder)
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder)

    # save  metrics as csv
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

    # Save the metrics as an image
    save_metrics_as_picture(metrics, output_folder / "final_metrics.png")

    print(f"Evaluación completada. Resultados en: {output_folder}")
    print("Métricas finales guardadas en 'final_metrics.png' y 'metrics.csv'")
