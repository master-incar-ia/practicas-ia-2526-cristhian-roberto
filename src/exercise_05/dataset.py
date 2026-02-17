from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.data = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

    def plot(self, filepath):
        plt.figure(figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                img = self.data[i * 10 + j][0].permute(1, 2, 0)
                # change axis 0 and 3
                plt.imshow(img, cmap=plt.cm.binary)
                plt.xlabel(self.data.classes[self.data[i * 10 + j][1]], fontsize=8)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.show()
        plt.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    # Data augmentation
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Revisar
            transforms.RandomCrop(32, padding=4),  # Revisar
            transforms.ToTensor(),
            #transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_train = CIFAR10Dataset("./data", train=True, transform=transform)
    dataset_test = CIFAR10Dataset("./data", train=False, transform=transform)
    dataset_val = CIFAR10Dataset("./data", train=False, transform=transform)
    print(f"Dataset de clasificación cargado: {len(dataset_train)} imágenes")
    print(f"First item: {dataset_train[0]}")
    print(f"Clases detectadas: {dataset_train.data.classes}")

    dataset_plot = CIFAR10Dataset("./data", train=True, transform=transforms.ToTensor())
    dataset_test = CIFAR10Dataset("./data", train=False, transform=transforms)
    dataset_plot.plot(output_folder / "plot_dataset_example.png")


# revisar para uso de función one_hot_encode en el modelo, ya que el dataset es de clasificación
# en el evaluate.py quiere que se muestre la matriz de confusión y las métricas de clasificación.
