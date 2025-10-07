"""Loader for CIFAR-10"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loader(batch_size: int, data_root: str, num_workers: int = 0):
    # define image transformations
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize to [-1, 1]
        ]
    )

    cifar10_train = datasets.CIFAR10(
        data_root,
        train=True,
        download=True,
        transform=transform,
    )

    cifar10_train = [img for img, _ in cifar10_train]  # remove labels

    # Dataloader
    loader = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
