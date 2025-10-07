"""Loader for MNIST dataset"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loader(batch_size: int, data_root: str, num_workers: int = 0):
    # define image transformations
    transform = transforms.Compose(
        [
            transforms.Pad(2),  # Pad 2 pixels on all sides to go from 28x28 to 32x32
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # Normalize to [-1, 1]
        ]
    )

    mnist_train = datasets.MNIST(
        data_root,
        train=True,
        download=True,
        transform=transform,
    )

    mnist_train = [img for img, _ in mnist_train]  # remove labels

    # Dataloader
    loader = DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
