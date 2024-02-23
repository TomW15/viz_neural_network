import numpy as np
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import typing as t

INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
DATA_FOLDER = Path(__file__).parent / "data/"


def fetch(train: bool = True, download: bool = False):

    if download:
        _ = MNIST(root=DATA_FOLDER, download=True)
    
    dataset = MNIST(root=DATA_FOLDER, train=train, transform=transforms.ToTensor())
    
    return dataset


def _load_mnist_dataset(train: bool = True) -> t.Tuple[torch.Tensor, torch.Tensor]:

    from viz_neural_network.models.mnist import dataset
    
    # Fetch data
    data = dataset.fetch(train=train)

    # Create lists for samples and labels
    X_list = [torch.Tensor(X) for (X, _) in data]
    y_list = [y for (_, y) in data]
    
    # Convert the data to proper numpy arrays and return
    return torch.Tensor(np.array(X_list)), torch.Tensor(y_list)


def load_mnist_dataset() -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    X, y = _load_mnist_dataset(train=True)
    X_test, y_test = _load_mnist_dataset(train=False)

    return X, y, X_test, y_test
