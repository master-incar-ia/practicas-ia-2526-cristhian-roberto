from pathlib import Path    # Para gestionar rutas de archivos de forma sencilla.
import matplotlib.pyplot as plt # Biblioteca para crear gráficos y visualizar los datos.
import numpy as np          # Biblioteca para realizar cálculos matemáticos, generar números aleatorios y manejo de arreglos.
import pandas as pd         # Biblioteca para manipulación y análisis de datos, especialmente con datos en tablas (estructuras tipo DataFrame).
import seaborn as sns   # Biblioteca para visualización de datos basada en matplotlib, que facilita la creación de gráficos estadísticos atractivos.
import torch       # Biblioteca principal de PyTorch para operaciones tensoriales y construcción de modelos de aprendizaje automático.
from torch.utils.data import Dataset  # La clase base de PyTorch que debemos heredar para crear nuestros propios conjuntos de datos. 

# Esta clase hereda de Dataset, lo que significa que PyTorch la reconocerá como una fuente de datos válida.
class NoisyRegressionDataset(Dataset):
    # Constructor del dataset sintético con ruido
    # Parámetros:
    # noise_std: Desviación estándar del ruido gaussiano añadido a los datos.
    # size: Número de muestras a generar.
    # seed: Semilla para la generación de números aleatorios, asegurando reproducibilidad.
    def __init__(
        self, noise_std=20, size=100, seed=42
    ):  
        np.random.seed(seed)  # Establece la semilla para la generación de números aleatorios.
        self.x = np.random.uniform(0, 100, size=(size,))    # Crea valores aleatorios entre 0 y 100.
        self.delta = np.random.normal(0, noise_std, size=(size,))   # Crea valores aleatorios siguiendo una "campana de Gauss" (distribución normal). Esto simula errores de medición de la vida real.
         # Define la relación lineal con el ruido añadido.
        self.y = 5 * self.x + 2 + self.delta

        # Create a DataFrame for visualization (Guarda x e y en una tabla para facilitar la creación de gráficos).
        df = pd.DataFrame(data=np.array([self.x, self.y]).transpose(), columns=["x", "y"])
        self.df = df

        # Reshape for PyTorch compatibility --> Transforma los datos de una lista simple a una columna (matriz de N×1). 
        # PyTorch requiere este formato para procesar los datos correctamente.
        self.x = self.x.reshape((-1, 1))
        self.y = self.y.reshape((-1, 1))

    # Método plot (Toma una ruta de archivo (filepath))
    def plot(self, filepath):
        # Crea un gráfico de dispersión (scatter plot) usando seaborn.
        ax = sns.scatterplot(self.df, x="x", y="y")
        # Añade título al gráfico.
        ax.set_title("Synthetic noisy data of y=5*x+2")
        # Guarda el gráfico en la ruta especificada.
        plt.savefig(filepath)
        # Muestra el gráfico.
        plt.show()

    # Método requerido por PyTorch Dataset: devuelve el tamaño del dataset.
    def __len__(self):
        return len(self.x)

    # Método requerido por PyTorch Dataset: devuelve un ítem (par x, y) dado un índice.
    # Es el método que usa PyTorch para obtener un ejemplo específico. 
    # Convierte los valores de Numpy a Tensores de PyTorch de tipo float32.
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )


if __name__ == "__main__":
    # Usa pathlib para crear una carpeta llamada outs donde se guardarán las imágenes.
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    # Crea la carpeta si no existe.
    output_folder.mkdir(exist_ok=True, parents=True)

    # Crea una instancia del dataset con 1000 muestras.
    dataset = NoisyRegressionDataset(size=1000)
    # Crea un segundo dataset mucho más ruidoso (dataset_ruidoso) aumentando noise_std a 100. 
    # Esto sirve para ver cómo el ruido dispersa los puntos.
    dataset_ruidoso = NoisyRegressionDataset(noise_std=100, size=1000)
    # Muestra en la consola el tamaño del dataset y cómo se ve el primer elemento (el par x,y).
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    # Generación de imágenes: Llama al método .plot() para ambos datasets, guardando los resultados como archivos .png.
    dataset.plot(output_folder / "plot_dataset_example.png")
    dataset_ruidoso.plot(output_folder / "plot_dataset_noisy_example.png")

    longitud_data = len(dataset)
    dataset.__len__()
    elemento_dataset = dataset[20]
