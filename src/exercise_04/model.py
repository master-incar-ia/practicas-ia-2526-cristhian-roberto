import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, input_dim=(3,32,32), hidden_dim=64, output_dim=10, apodo=None):
        super().__init__()
        self.apodo = apodo
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, use_activation=False):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = NonlinearRegressor(1, 64, 1, "MiModeloNoLineal_1")
    # model2 = NonlinearRegressor(1, 128, 1, "MiModeloNoLineal_2")

    x = torch.tensor([1.0])
    print(model.forward(x))
    # print(model2.forward(x))

    # 1. Instanciamos el modelo
    # input_dim = 1
    # hidden_dim = 64
    # output_dim = 1
    # model = NonlinearRegressor(input_dim, hidden_dim, output_dim)

    # print("--- Análisis del Modelo para Regresión Cuadrática ---")

    # # 2. Mostrar la arquitectura
    # print(f"\nEstructura del modelo:\n{model}")

    # # 3. Cálculo de parámetros (Muestra la complejidad del modelo)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"\nTotal de parámetros entrenables: {total_params}")
    # print(
    #     "Nota: Un modelo lineal simple solo tendría 2 parámetros. "
    #     "Este modelo tiene capacidad para aprender curvaturas."
    # )

    # # 4. Prueba de Dimensiones (Sanity Check)
    # # Simulamos un 'batch' de 5 datos de entrada
    # dummy_input = torch.randn(5, 1)
    # dummy_output = model(dummy_input)

    # print(f"\nPrueba de dimensiones:")
    # print(f"Entrada: {dummy_input.shape} -> Salida: {dummy_output.shape}")
    # if dummy_input.shape == dummy_output.shape:
    #     print("✅ Las dimensiones de salida son correctas para una regresión.")

    # # 5. Prueba de No-Linealidad (Demostración de efectividad)
    # # Si el modelo fuera lineal, la diferencia entre f(1), f(2) y f(3) sería constante.
    # # Al ser no-lineal (ReLU), el modelo tiene potencial de ajustarse a la parábola.
    # test_points = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    # with torch.no_grad():
    #     results = model(test_points)

    # diff1 = results[1] - results[0]
    # diff2 = results[2] - results[1]

    # print(f"\nVerificación de capacidad no-lineal:")
    # print(f"f(1) = {results[0].item():.4f}")
    # print(f"f(2) = {results[1].item():.4f}")
    # print(f"f(3) = {results[2].item():.4f}")

    # # Si las diferencias no son iguales, el modelo es capaz de romper la linealidad
    # if not torch.isclose(diff1, diff2):
    #     print("✅ El modelo detecta cambios no-lineales (gracias a ReLU).")
    # else:
    #     print("⚠️ El modelo se comporta linealmente en este rango (común antes de entrenar).")

    # print("\n--- Listo para iniciar el entrenamiento cuadrático ---")
