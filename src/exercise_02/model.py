import torch
import torch.nn as nn

class SimplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
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
        self.activation_relu = nn.ReLU()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)
        x1 = self.activation_relu(x1)
        x2 = self.fc2(x1)

        if use_activation:
            x2 = self.activation(x2)
        return x2


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
