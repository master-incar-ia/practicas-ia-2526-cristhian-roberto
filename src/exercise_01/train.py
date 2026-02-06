from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim  # Biblioteca de optimización de PyTorch. Contiene los algoritmos de optimización (como AdamW) que ajustan los pesos de la red.
from torch.utils.data import (  # Herramientas para cargar datos y dividir conjuntos de datos.
    DataLoader,  # Una herramienta de PyTorch que divide los datos en grupos pequeños (batches) para que la computadora pueda procesarlos por partes.
    random_split,  # Una función que divide un conjunto de datos en partes más pequeñas de manera aleatoria.
)
from tqdm import tqdm  # Biblioteca para mostrar barras de progreso en loops.

from .dataset import NoisyRegressionDataset
from .model import MultiLayerPerceptron, SimplePerceptron


# Función para obtener el dispositivo (CPU o GPU) según la disponibilidad y preferencia del usuario.
def get_device(force: str = "auto") -> torch.device:
    """Return a torch.device based on the `force` option.

    force: 'auto'|'cpu'|'cuda' - when 'auto' will pick cuda if available.
    """
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Función principal para entrenar el modelo
# Parámetros:
# output_folder: Ruta donde se guardarán los resultados.
# device: Dispositivo (CPU o GPU) donde se realizará el entrenamiento.
def train_model(output_folder: Path, device: torch.device):
    # Create an instance of the dataset
    dataset = NoisyRegressionDataset(size=10000)

    # Split the dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for the datasets
    # shuffle=True (mezcla los datos en cada vuelta), lo cual es fundamental para que el modelo
    # no aprenda el orden de los datos, sino la lógica detrás de ellos.
    pin_memory = True if device.type == "cuda" else False  # Para optimizar la transferencia a GPU
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, pin_memory=pin_memory)

    # Define the model, loss function, and optimizer
    input_dim = 1
    output_dim = 1
    model = MultiLayerPerceptron(input_dim, output_dim).to(device)
    # Es la métrica de error. Como es una regresión, usa el Error Cuadrático Medio (MSE).
    criterion = nn.MSELoss()
    #  Es el algoritmo que mueve los pesos del modelo para reducir el error. lr=0.0001 es la velocidad de aprendizaje.
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # Training loop with validation and saving best weights
    num_epochs = (
        100  # Número de veces que el modelo verá todo el conjunto de datos de entrenamiento.
    )
    best_val_loss = float("inf")  # Inicializa la mejor función pérdida de validación con infinito.
    best_model_path = (
        output_folder / "best_model.pth"
    )  # Ruta para guardar los mejores pesos del modelo.

    train_losses = []
    val_losses = []

    # Loop de entrenamiento
    for epoch in tqdm(range(num_epochs)):
        # Training step: Pone el modelo en modo entrenamiento (activa dropout, batchnorm, etc.)
        # Esto es importante porque algunas capas funcionan de manera diferente durante el entrenamiento y la evaluación.
        model.train()
        # Inicializa la pérdida de entrenamiento en 0 para acumularla durante las iteraciones.
        train_loss = 0
        # Itera sobre los batches del DataLoader de entrenamiento.
        for inputs, targets in train_loader:
            # Forward pass: calcula las predicciones del modelo y la pérdida.
            inputs_cuda = inputs.to(
                device
            )  # Mueve los datos de entrada al dispositivo (CPU o GPU).
            targets_cuda = targets.to(
                device
            )  # Mueve los datos objetivo al dispositivo (CPU o GPU).
            outputs = model(
                inputs_cuda, use_activation=False
            )  # Obtiene las predicciones del modelo.
            loss = criterion(
                outputs, targets_cuda
            )  # Calcula la pérdida entre las predicciones y los objetivos.

            train_loss += loss.item()  # Acumula la pérdida de entrenamiento.

            # Backward pass and optimization
            optimizer.zero_grad()  # Limpia los gradientes acumulados.
            loss.backward()  # Calcula los gradientes del Loss con respecto a los pesos del modelo.
            optimizer.step()  # Actualiza los pesos del modelo usando los gradientes calculados.

        train_loss /= len(
            train_loader
        )  # Promedia la pérdida de entrenamiento sobre todos los batches.
        train_losses.append(train_loss)  # Guarda la pérdida de entrenamiento para este epoch.

        # Validation step: Pone el modelo en modo evaluación (desactiva dropout, batchnorm, etc.)
        # Esto es importante para evaluar el rendimiento real del modelo. No se calculan gradientes durante la
        # validación para ahorrar memoria y computación.
        model.eval()
        val_loss = 0
        # Desactiva el cálculo de gradientes.
        # Esto es crucial para la validación, ya que no se necesita actualizar los pesos del modelo.
        with torch.no_grad():
            # Itera sobre los batches del DataLoader de validación.
            for inputs, targets in val_loader:
                inputs_cuda = inputs.to(
                    device
                )  # Mueve los datos de entrada al dispositivo (CPU o GPU).
                targets_cuda = targets.to(
                    device
                )  # Mueve los datos objetivo al dispositivo (CPU o GPU).
                outputs = model(
                    inputs_cuda, use_activation=False
                )  # Obtiene las predicciones del modelo.
                loss = criterion(
                    outputs, targets_cuda
                )  # Calcula la pérdida entre las predicciones y los objetivos.
                val_loss += loss.item()  # Acumula la pérdida de validación.

        val_loss /= len(val_loader)  # Promedia la pérdida de validación sobre todos los batches.
        val_losses.append(val_loss)  # Guarda la pérdida de validación para este epoch.

        if (
            val_loss < best_val_loss
        ):  # Si la pérdida de validación es la mejor hasta ahora, guarda los pesos del modelo.
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )
    # Final print of the best validation loss
    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))  # Crea una figura de 10x5 pulgadas.
    plt.plot(
        range(num_epochs), train_losses, label="Train Loss"
    )  # Grafica la pérdida de entrenamiento.
    plt.plot(
        range(num_epochs), val_losses, label="Validation Loss"
    )  # Grafica la pérdida de validación.
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save the plot to the outs/ folder
    plt.savefig(output_folder / "loss_plot.png")


if __name__ == "__main__":
    # Create output folder based on file folder
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")  # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")
    # Set the seed for reproducibility
    torch.manual_seed(42)
    train_model(output_folder, device=device)
