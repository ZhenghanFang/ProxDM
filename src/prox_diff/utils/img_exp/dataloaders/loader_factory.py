from .cifar10_loader import get_cifar10_loader
from .mnist_loader import get_mnist_loader


def get_loader(dataset_name, batch_size, num_workers):
    if dataset_name == "cifar10":
        loader = get_cifar10_loader(
            batch_size, data_root="data/cifar10", num_workers=num_workers
        )
    elif dataset_name == "mnist":
        loader = get_mnist_loader(
            batch_size, data_root="data/mnist", num_workers=num_workers
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return loader
