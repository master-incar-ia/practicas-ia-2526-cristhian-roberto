import torch
import torch.nn as nn
import torch.nn.functional as F


# MODELO FULLY CONNECTED (Denso)
class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=1024, output_dim=10):
        super().__init__()
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            # Capa 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # <--- NUEVO: Estabiliza los gradientes
            nn.ReLU(),
            nn.Dropout(0.3),  # <--- Aumentado: Evita el sobreajuste
            # Capa 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Capa 3
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            # Capa de salida
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)


if __name__ == "__main__":
    # --- PRUEBA DE SANIDAD: MODELO FULLY CONNECTED (FC) ---
    print("--- Verificando Modelo Fully Connected ---")
    # Instanciamos el modelo FC
    model_fc = FullyConnectedClassifier(input_dim=3 * 32 * 32, hidden_dim=512, output_dim=10)

    # Creamos un batch de prueba (ej: 5 imágenes aleatorias)
    dummy_batch_fc = torch.randn(5, 3, 32, 32)

    # Paso hacia adelante (Forward pass)
    output_fc = model_fc(dummy_batch_fc)

    print(f"Entrada FC (batch de 5): {dummy_batch_fc.shape}")
    print(f"Salida FC (logits): {output_fc.shape}")  # Esperado: [5, 10]

    # Verificación de parámetros
    params_fc = sum(p.numel() for p in model_fc.parameters() if p.requires_grad)
    print(f"Parámetros entrenables FC: {params_fc:,}")

    # --- COMPROBACIÓN LÓGICA DE CLASIFICACIÓN ---
    # Probamos a pasar la salida por una función Softmax para ver "probabilidades"
    probabilities = torch.softmax(output_fc, dim=1)
    sum_probs = torch.sum(probabilities[0])

    print(f"\nVerificación de Softmax:")
    print(f"Suma de probabilidades de la primera imagen: {sum_probs.item():.2f}")
    if torch.isclose(sum_probs, torch.tensor(1.0)):
        print("✅ Correcto: Las salidas representan una distribución de probabilidad.")

    # Verificamos qué clase "predice" (la de mayor valor)
    predicted_class = torch.argmax(probabilities[0])
    print(f"Clase predicha para imagen 1 (sin entrenar): {predicted_class.item()}")
