import torch    # Importa la librería PyTorch para construir y entrenar modelos de aprendizaje automático.
import torch.nn as nn   # Importa el módulo de redes neuronales de PyTorch, que proporciona herramientas para crear modelos de redes neuronales (capas lineales, funciones de activación, etc.).

# Definición de una clase SimplePerceptron que hereda de nn.Module (la clase base para todos los modelos de PyTorch).
class SimplePerceptron(nn.Module):
    # Constructor que inicializa las capas del perceptrón.
    # Parámetros:
    # input_dim: Dimensión de la entrada (número de características).
    # output_dim: Dimensión de la salida (número de clases o valores a predecir).
    def __init__(self, input_dim, output_dim):
        # Llama al constructor de la clase base nn.Module.
        super().__init__()
        # Define una capa lineal (fully connected) que conecta la entrada con la salida.
        self.fc = nn.Linear(input_dim, output_dim)
        # Define una función de activación (aquí es la identidad, es decir, no hace nada).
        self.activation = nn.Identity()
        # Define una función de activación ReLU (Rectified Linear Unit), que convierte todos los valores negativos en cero.
        self.activation_relu = nn.ReLU()

    # Método forward que define cómo se propaga la información a través del modelo.
    # Parámetros:
    # x: Entrada al modelo.
    # use_activation: Booleano que indica si se debe aplicar la función de activación al final.
    # Retorna la salida del modelo después de aplicar las capas y la activación.
    def forward(self, x, use_activation=True):
        x = self.fc(x)
        x = self.activation_relu(x) # Aplicación de la función de activación ReLU
        return x 

# Definición de una clase MultiLayerPerceptron que hereda de nn.Module.
class MultiLayerPerceptron(nn.Module):
    # Constructor que inicializa las capas del perceptrón multicapa.
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo):
        super().__init__()
        # Define la primera capa lineal que conecta la entrada con la capa oculta.
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons)
        # Define la segunda capa lineal que conecta la capa oculta con la salida.
        self.fc2 = nn.Linear(num_hidden_neurons, output_dim)
        # Define una función de activación (identidad).
        self.activation = nn.Identity()
        # Define una función de activación ReLU (Rectified Linear Unit), que convierte todos los valores negativos en cero.
        self.activation_relu = nn.ReLU()
        self.apodo = apodo

    # Método forward que define cómo se propaga la información a través del modelo.
    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)

        x1 = self.activation_relu(x1)
        x2 = self.fc2(x1)
        # Define una función de activación ReLU (Rectified Linear Unit)
        if use_activation:
            x2 = self.activation(x2)
        return x2


if __name__ == "__main__":
    # Recibe 1 dato de entrada (como la x del dataset anterior). Tiene 1 dato de salida (como la y del dataset).
    # Usa 2 neuronas en su capa oculta. Es un modelo pequeño diseñado para regresión simple.
    model1 = MultiLayerPerceptron(1, 1, 2, "MiModeloSencillo")
    # Un modelo más grande con 1000 datos de entrada, 2 datos de salida y 16 neuronas en su capa oculta.
    model2 = MultiLayerPerceptron(1000, 2, 16, "MiModeloDeDesfibrilador")

    # Prueba rápida del modelo con un dato de entrada.
    x = torch.tensor([1.0])
    print(model1.forward(x))
    pass
    # print(model)
    # x = torch.tensor([1.0])
    # print(model(x))
    # pass
