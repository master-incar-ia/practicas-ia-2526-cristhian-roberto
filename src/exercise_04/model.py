import torch
import torch.nn as nn


class SimplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Identity()
        self.activation_relu = nn.ReLU()

    def forward(self, x, use_activation=True):
        x = self.fc(x)
        x = self.activation_relu(x)
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons)
        self.fc2 = nn.Linear(num_hidden_neurons, output_dim)
        self.activation = nn.Identity()
        self.activation_tanh = nn.Tanh()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)
        x1 = self.activation_tanh(x1)
        x2 = self.fc2(x1)

        if use_activation:
            x2 = self.activation(x2)
        return x2


class DoubleMultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons)
        self.fc2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc3 = nn.Linear(num_hidden_neurons, output_dim)

        self.activation_tanh = nn.Tanh()
        self.activation = nn.Identity()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x = self.fc1(x)
        x = self.activation_tanh(x)
        x = self.fc2(x)
        x = self.activation_tanh(x)
        x = self.fc3(x)

        if use_activation:
            x = self.activation(x)
        return x


class DoubleConvNet(nn.Module):
    def __init__(self, in_channels, output_dim, num_filters, dropout_p=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout_conv = nn.Dropout2d(p=dropout_p)

        self.dropout_fc = nn.Dropout(p=0.3)

        self.fc = nn.Linear(num_filters * 8 * 8, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)

        x = self.dropout_fc(x)
        x = self.fc(x)

        return x


# MODELO CNN (Optimizado para imágenes)
class CNNClassifier(nn.Module):
    def __init__(self, output_dim=10):
        super().__init__()
        # Bloque 1: 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Capa extra para profundidad
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        # Bloque 2: 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )

        # Bloque 3: 8x8 -> 4x4 (NUEVO)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
        )

        # Clasificador
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),  # El tamaño cambia de 8x8 a 4x4
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc_layer(x)
        return x

if __name__ == "__main__":
    model1 = MultiLayerPerceptron(1, 1, 2, "MiModeloSencillo")
    model2 = MultiLayerPerceptron(1000, 2, 16, "MiModeloDeDesfibrilador")

    x = torch.tensor([1.0])
    print(model1.forward(x))
    pass
    # print(model)
    # x = torch.tensor([1.0])
    # print(model(x))
    # pass
